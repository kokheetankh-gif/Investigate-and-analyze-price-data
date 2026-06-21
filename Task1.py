import sys
import io
import warnings
import subprocess
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# GitHub source
_GITHUB_URL = (
    "https://raw.githubusercontent.com/"
    "kokheetankh-gif/Investigate-and-analyze-price-data/main/Nat_Gas.csv"
)


def _fetch_data(url: str = _GITHUB_URL) -> pd.DataFrame:
    """Download CSV from GitHub and return a clean DataFrame."""
    try:
        import urllib.request
        with urllib.request.urlopen(url) as resp:
            raw = resp.read().decode("utf-8")
    except Exception:
        # Fallback: use curl (works in restricted envs)
        result = subprocess.run(["curl", "-s", url], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Could not fetch data from {url}")
        raw = result.stdout

    df = pd.read_csv(io.StringIO(raw))
    df["date"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month
    return df


# Fit the model once at import time 
def _build_model():
    df = _fetch_data()

    # Reference point = first data date (31 Oct 2020)
    t0 = df["date"].min()

    def _months_since(d):
        ts = pd.Timestamp(d)
        return (ts.year - t0.year) * 12 + (ts.month - t0.month)

    df["t"] = df["date"].apply(_months_since)
    trend_coef = np.polyfit(df["t"], df["Prices"], 1)   # [slope, intercept]
    trend_fn = np.poly1d(trend_coef)
    df["trend"] = trend_fn(df["t"])
    df["residual"] = df["Prices"] - df["trend"]

    seasonal_adj = df.groupby("month")["residual"].mean()   # 12 monthly averages
    resid_std = df["residual"].std()
    last_date = df["date"].max()

    return t0, trend_fn, seasonal_adj, resid_std, last_date, trend_coef


print("Fetching data from GitHub...", end=" ", flush=True)
_T0, _TREND_FN, _SEASONAL_ADJ, _RESID_STD, _LAST_DATE, _TREND_COEF = _build_model()
print("done.\n")


# Public API 
def estimate_price(input_date, return_bounds: bool = False):
    """
    Estimate the natural gas purchase price for any calendar date.

    Parameters:
        input_date : str | datetime | date | pd.Timestamp
            Target date. Common string formats accepted:
            '2025-06-30'  |  '6/30/2025'  |  '30 Jun 2025'
            Only the year and month matter; day is ignored.

        return_bounds : bool, default False
            If True, returns a ±2σ uncertainty interval as well.

    Returns: 
        float
            Estimated price in $/MMBtu, rounded to 2 dp.

        tuple (float, float, float)  — only when return_bounds=True
            (point_estimate, lower_bound, upper_bound)
    """
    ts = pd.Timestamp(input_date)
    t_val = (ts.year - _T0.year) * 12 + (ts.month - _T0.month)
    month = ts.month

    price = round(float(_TREND_FN(t_val) + _SEASONAL_ADJ[month]), 2)

    if return_bounds:
        lo = round(price - 2 * _RESID_STD, 2)
        hi = round(price + 2 * _RESID_STD, 2)
        return price, lo, hi

    return price


# CLI / demo 
def _demo():
    from dateutil.relativedelta import relativedelta

    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    print("=" * 65)
    print("  Natural Gas Price Estimator")
    print(f"  Data source : {_GITHUB_URL}")
    print(f"  Data window : Oct 2020 – {_LAST_DATE.strftime('%b %Y')}")
    print( "  Model       : Linear trend + seasonal monthly adjustment")
    print(f"  Trend slope : {_TREND_COEF[0]:+.4f} $/month"
          f"  ({_TREND_COEF[0]*12:+.2f} $/year)")
    print(f"  Residual σ  : {_RESID_STD:.4f}  →  ±2σ ≈ ±{2*_RESID_STD:.2f}")
    print("=" * 65)

    print("\n── Historical spot-checks (observed vs estimated) ──")
    spot = [("2021-03-31",10.90),("2022-06-30",10.40),
            ("2023-01-31",12.10),("2024-06-30",11.50)]
    print(f"  {'Date':<13}  {'Observed':>10}  {'Estimated':>10}  {'Error':>8}")
    print("  " + "─" * 47)
    for d, obs in spot:
        est = estimate_price(d)
        print(f"  {d:<13}  {obs:>10.2f}  {est:>10.2f}  {est-obs:>+8.2f}")

    print("\n── 12-Month Extrapolation ──")
    print(f"  {'Month':<10}  {'Estimate':>10}  {'Lower (−2σ)':>12}  {'Upper (+2σ)':>12}")
    print("  " + "─" * 50)
    for i in range(1, 13):
        d = _LAST_DATE + relativedelta(months=i)
        p, lo, hi = estimate_price(d, return_bounds=True)
        print(f"  {d.strftime('%b %Y'):<10}  {p:>10.2f}  {lo:>12.2f}  {hi:>12.2f}")

    print("\n── Seasonal Profile ──")
    print(f"  {'Month':<6}  {'Avg Deviation':>15}")
    print("  " + "─" * 25)
    for m, name in enumerate(MONTHS, 1):
        adj = _SEASONAL_ADJ[m]
        bar = ("▲" if adj >= 0 else "▼") + " " + "█" * int(abs(adj) / 0.1)
        print(f"  {name:<6}  {adj:>+8.2f}   {bar}")
    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw = sys.argv[1]
        try:
            p, lo, hi = estimate_price(raw, return_bounds=True)
            print(f"Date     : {pd.Timestamp(raw).strftime('%d %b %Y')}")
            print(f"Estimate : ${p:.2f} /MMBtu")
            print(f"Range    : ${lo:.2f} – ${hi:.2f}  (±2σ interval)")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        _demo()