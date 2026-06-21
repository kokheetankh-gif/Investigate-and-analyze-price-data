"""
FICO Score Quantization — Optimal Bucket Boundaries
JPMorgan Chase Quantitative Research — Task 4

Problem
-------
Map integer FICO scores (300–850) into a fixed number of ordered buckets
(ratings) so that:
  Rating 1  →  best credit (high FICO, low default probability)
  Rating n  →  worst credit (low FICO, high default probability)

Two optimisation criteria are implemented:

  1. MSE minimisation  — minimise the mean squared error between each
     borrower's FICO score and their bucket's representative value (mean).
     Treats bucketing as a signal-approximation problem.

  2. Log-likelihood maximisation — maximise:
         LL = Σ_i [ k_i · ln(p_i) + (n_i - k_i) · ln(1 - p_i) ]
     where n_i = records in bucket i, k_i = defaults, p_i = k_i/n_i.
     This rewards buckets that are internally homogeneous in default rate
     and penalises degenerate splits.  Solved via dynamic programming.

Both are solved with an exact O(B · S²) dynamic programming algorithm
(B = number of buckets, S = number of distinct FICO scores).

Output
------
  - Boundary arrays for each method and each n_buckets
  - A rating map function:  fico_score → rating (1 = best, n = worst)
  - Printed comparison table and bucket statistics
"""

import io
import os
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

_GITHUB_URL = (
    "https://raw.githubusercontent.com/"
    "kokheetankh-gif/Investigate-and-analyze-price-data/main/"
    "Task%203%20and%204_Loan_Data.csv"
)

# 1.  DATA LOADING
def _load_data(csv_path=None) -> pd.DataFrame:
    if csv_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [
            os.path.join(script_dir, "Task_3_and_4_Loan_Data.csv"),
            os.path.join(os.getcwd(),  "Task_3_and_4_Loan_Data.csv"),
        ]:
            if os.path.exists(candidate):
                csv_path = candidate
                break

    if csv_path and os.path.exists(str(csv_path)):
        return pd.read_csv(csv_path)

    print("  [info] Loan data not found locally — fetching from GitHub...")
    try:
        import urllib.request
        with urllib.request.urlopen(_GITHUB_URL) as r:
            raw = r.read().decode("utf-8")
    except Exception:
        import subprocess
        raw = subprocess.run(
            ["curl", "-s", _GITHUB_URL], capture_output=True, text=True
        ).stdout
    return pd.read_csv(io.StringIO(raw))


def _build_fico_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per distinct FICO score.
    Returns DataFrame with columns: fico_score, n (count), k (defaults), p (default rate).
    Sorted ascending by FICO score.
    """
    tbl = (
        df.groupby("fico_score")
          .agg(n=("default", "count"), k=("default", "sum"))
          .reset_index()
          .sort_values("fico_score")
          .reset_index(drop=True)
    )
    tbl["p"] = tbl["k"] / tbl["n"]
    return tbl

# 2.  COST FUNCTIONS  (precomputed for all contiguous sub-arrays)
def _precompute_mse_costs(tbl: pd.DataFrame) -> np.ndarray:
    """
    cost_mse[i][j] = MSE cost of placing FICO-score indices i..j in one bucket.

    Within a bucket the representative value is the weighted mean FICO score,
    and the cost is the weighted sum of squared deviations.
    """
    S = len(tbl)
    scores = tbl["fico_score"].values.astype(float)
    ns = tbl["n"].values.astype(float)

    # Prefix sums for weighted computations
    w_sum = np.zeros(S + 1)   # Σ n_i
    wx_sum = np.zeros(S + 1)   # Σ n_i * x_i
    wx2_sum = np.zeros(S + 1)   # Σ n_i * x_i²

    for i in range(S):
        w_sum[i+1] = w_sum[i]   + ns[i]
        wx_sum[i+1] = wx_sum[i]  + ns[i] * scores[i]
        wx2_sum[i+1] = wx2_sum[i] + ns[i] * scores[i]**2

    cost = np.full((S, S), np.inf)
    for i in range(S):
        for j in range(i, S):
            w = w_sum[j+1]   - w_sum[i]
            wx = wx_sum[j+1]  - wx_sum[i]
            wx2 = wx2_sum[j+1] - wx2_sum[i]
            if w > 0:
                mean  = wx / w
                # MSE = (1/N) * Σ n_i*(x_i - mean)²
                #     = (1/N) * (Σ n_i*x_i² - N*mean²)
                cost[i][j] = (wx2 - w * mean**2) / w
    return cost


def _precompute_ll_costs(tbl: pd.DataFrame) -> np.ndarray:
    """
    cost_ll[i][j] = NEGATIVE log-likelihood of placing indices i..j in one bucket.
    (Negative so we can minimise instead of maximise.)

    LL_bucket = k * ln(p) + (n-k) * ln(1-p)   where p = k/n

    Edge cases:
      - k=0  (no defaults):   LL = 0  (only the (n-k)*ln(1-p) term, p→0 gives 0)
      - k=n  (all default):   LL = 0  (only the k*ln(p) term, p→1 gives 0)
      - k=0 or k=n handled by clipping p away from 0/1 with a small epsilon.
    """
    S = len(tbl)
    ns = tbl["n"].values.astype(float)
    ks = tbl["k"].values.astype(float)

    # Prefix sums
    n_pre = np.zeros(S + 1)
    k_pre = np.zeros(S + 1)
    for i in range(S):
        n_pre[i+1] = n_pre[i] + ns[i]
        k_pre[i+1] = k_pre[i] + ks[i]

    cost = np.full((S, S), np.inf)
    EPS = 1e-10
    for i in range(S):
        for j in range(i, S):
            n_b = n_pre[j+1] - n_pre[i]
            k_b = k_pre[j+1] - k_pre[i]
            if n_b > 0:
                p = np.clip(k_b / n_b, EPS, 1 - EPS)
                ll = k_b * np.log(p) + (n_b - k_b) * np.log(1 - p)
                cost[i][j] = -ll   # negate → minimise
    return cost

# 3.  DYNAMIC PROGRAMMING SOLVER
def _dp_solve(cost: np.ndarray, n_buckets: int) -> list:
    """
    Exact O(B * S²) DP to find the optimal split of S items into n_buckets
    contiguous segments minimising total cost.

    Returns a list of (start_idx, end_idx) pairs (inclusive) for each bucket,
    ordered from lowest FICO index to highest.
    """
    S = cost.shape[0]
    B = n_buckets

    INF = float("inf")

    # dp[b][j] = minimum cost using b buckets covering items 0..j
    dp   = np.full((B + 1, S), INF)
    # split[b][j] = best split point i for dp[b][j]  (bucket b covers i..j)
    split = np.full((B + 1, S), -1, dtype=int)

    # Base case: 1 bucket covering 0..j
    for j in range(S):
        dp[1][j]    = cost[0][j]
        split[1][j] = 0

    # Fill DP table
    for b in range(2, B + 1):
        for j in range(b - 1, S):
            for i in range(b - 1, j + 1):
                c = dp[b-1][i-1] + cost[i][j] if i > 0 else INF
                if c < dp[b][j]:
                    dp[b][j]    = c
                    split[b][j] = i

    # Traceback
    segments = []
    j = S - 1
    for b in range(B, 0, -1):
        i = split[b][j]
        segments.append((i, j))
        j = i - 1
    segments.reverse()
    return segments

# 4.  PUBLIC API
def find_buckets(
    n_buckets: int = 5,
    method: str = "log_likelihood",
    csv_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Find optimal FICO score bucket boundaries.

    Parameters:
        n_buckets : int
            Number of rating buckets (default: 5).
        method    : str
            'log_likelihood' (default) or 'mse'.
        csv_path  : str or None
            Path to Loan_Data.csv; auto-detects if None.
        verbose   : bool
            Print bucket summary if True.

    Returns:
        dict with keys:
            boundaries     : list of (n_buckets + 1) FICO score boundaries
                            e.g. [408, 560, 620, 670, 720, 850]
            bucket_stats   : DataFrame with per-bucket n, k, p, rating
            rating_map     : callable  fico_score (int) → rating (1=best, n=worst)
            total_cost     : float  (MSE or negative log-likelihood)
    """
    if method not in ("log_likelihood", "mse"):
        raise ValueError("method must be 'log_likelihood' or 'mse'")

    df  = _load_data(csv_path)
    tbl = _build_fico_table(df)
    S = len(tbl)
    scores = tbl["fico_score"].values

    # Precompute cost matrix
    if method == "mse":
        cost = _precompute_mse_costs(tbl)
    else:
        cost = _precompute_ll_costs(tbl)

    # Solve DP
    segments = _dp_solve(cost, n_buckets)

    # Build boundary list  [lower_1, lower_2, ..., lower_n, upper_n]
    boundaries = []
    for idx, (i, j) in enumerate(segments):
        boundaries.append(int(scores[i]))
    boundaries.append(int(scores[segments[-1][1]]))

    # Build bucket stats — ratings assigned highest FICO = rating 1 (best)
    rows = []
    for rating_rev, (i, j) in enumerate(segments):
        n_b = tbl["n"].iloc[i:j+1].sum()
        k_b = tbl["k"].iloc[i:j+1].sum()
        p_b = k_b / n_b if n_b > 0 else 0.0
        rows.append({
            "fico_lower": int(scores[i]),
            "fico_upper": int(scores[j]),
            "n": int(n_b),
            "k": int(k_b),
            "default_rate": round(p_b, 4),
        })

    # Reverse so that rating 1 = best (lowest default, highest FICO)
    rows = rows[::-1]
    for r, row in enumerate(rows):
        row["rating"] = r + 1

    stats_df = pd.DataFrame(rows)[["rating", "fico_lower", "fico_upper", "n", "k", "default_rate"]]

    # Rating map function
    # Boundaries after reversal: rating 1 is highest FICO segment
    # We build a lookup: for each unique FICO score → rating
    fico_to_rating = {}
    for _, row in stats_df.iterrows():
        sub = tbl[(tbl["fico_score"] >= row["fico_lower"]) &
                  (tbl["fico_score"] <= row["fico_upper"])]
        for fs in sub["fico_score"]:
            fico_to_rating[int(fs)] = int(row["rating"])

    def rating_map(fico_score: int) -> int:
        """Map a FICO score to a rating (1=best credit, n=worst credit)."""
        fs = int(fico_score)
        if fs in fico_to_rating:
            return fico_to_rating[fs]
        # For unseen scores, assign based on bucket boundaries
        for _, row in stats_df.iterrows():
            if row["fico_lower"] <= fs <= row["fico_upper"]:
                return int(row["rating"])
        # Outside range — clamp
        return 1 if fs > stats_df["fico_upper"].max() else int(n_buckets)

    total_cost = sum(cost[i][j] for i, j in segments)

    if verbose:
        meth_label = "Log-Likelihood" if method == "log_likelihood" else "MSE"
        W = 66
        print(f"\n{'='*W}")
        print(f"  FICO BUCKETING — {meth_label.upper()} — {n_buckets} BUCKETS")
        print(f"{'='*W}")
        print(f"  Boundaries: {boundaries}")
        print(f"  Total cost: {total_cost:.4f}")
        print()
        print(f"  {'Rating':<8} {'FICO Range':<18} {'n':>6} {'defaults':>9} {'Default Rate':>13}")
        print("  " + "─" * (W - 2))
        for _, row in stats_df.iterrows():
            rng = f"{row['fico_lower']}–{row['fico_upper']}"
            print(f"  {row['rating']:<8} {rng:<18} {row['n']:>6,} "
                  f"{row['k']:>9,} {row['default_rate']:>12.1%}")
        print(f"{'='*W}\n")

    return {
        "boundaries":   boundaries,
        "bucket_stats": stats_df,
        "rating_map":   rating_map,
        "total_cost":   total_cost,
    }


def compare_methods(n_buckets: int = 5, csv_path: str = None):
    """
    Run both methods and print a side-by-side comparison.
    Returns (ll_result, mse_result).
    """
    print(f"\n{'='*66}")
    print(f"  COMPARING METHODS FOR {n_buckets} BUCKETS")
    print(f"{'='*66}")
    ll_result  = find_buckets(n_buckets=n_buckets, method="log_likelihood",
                              csv_path=csv_path, verbose=True)
    mse_result = find_buckets(n_buckets=n_buckets, method="mse",
                              csv_path=csv_path, verbose=True)
    return ll_result, mse_result



# 5.  DEMO
if __name__ == "__main__":

    CSV = None  # auto-detect: script folder → cwd → GitHub

    # Run both methods for 5 buckets
    ll_5, mse_5 = compare_methods(n_buckets=5, csv_path=CSV)

    # Show that the rating map works
    print("=" * 66)
    print("  RATING MAP EXAMPLES (Log-Likelihood, 5 buckets)")
    print("=" * 66)
    rating_map = ll_5["rating_map"]
    test_ficos = [420, 520, 580, 620, 650, 680, 710, 750, 800, 850]
    print(f"  {'FICO Score':<14} {'Rating':>8}  {'Interpretation'}")
    print("  " + "─" * 50)
    labels = {1:"Best credit", 2:"Good credit", 3:"Fair credit",
              4:"Poor credit", 5:"Worst credit"}
    for f in test_ficos:
        r = rating_map(f)
        print(f"  {f:<14} {r:>8}  {labels.get(r,'')}")
    print()

    # Try 3 and 10 buckets with log-likelihood
    for nb in [3, 10]:
        find_buckets(n_buckets=nb, method="log_likelihood",
                     csv_path=CSV, verbose=True)