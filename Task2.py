import os
import sys
import io
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_GITHUB_URL = (
    "https://raw.githubusercontent.com/"
    "kokheetankh-gif/Investigate-and-analyze-price-data/main/Nat_Gas.csv"
)

def _resolve_csv(csv_path):
    """
    Return a readable path or URL for the CSV.
    Priority: explicit arg → script dir → cwd → GitHub URL.
    """
    if csv_path is not None:
        return csv_path

    # Try folder containing this .py file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(script_dir, "Nat_Gas.csv")
    if os.path.exists(local):
        return local

    # Try current working directory
    cwd_path = os.path.join(os.getcwd(), "Nat_Gas.csv")
    if os.path.exists(cwd_path):
        return cwd_path

    # Fallback: fetch from GitHub
    print("  [info] Nat_Gas.csv not found locally — fetching from GitHub...")
    return _GITHUB_URL

# 1.  PRICE ESTIMATOR
def _load_csv(source: str) -> pd.DataFrame:
    """Load CSV from a local path or a URL."""
    if source.startswith("http"):
        try:
            import urllib.request
            with urllib.request.urlopen(source) as r:
                raw = r.read().decode("utf-8")
        except Exception:
            import subprocess
            result = subprocess.run(["curl", "-s", source], capture_output=True, text=True)
            raw = result.stdout
        return pd.read_csv(io.StringIO(raw))
    return pd.read_csv(source)


def _build_price_model(csv_source: str):
    """
    Fit linear-trend + sinusoidal-seasonal model from the CSV.
    Returns a callable  estimate_price(date_like) -> float.
    """
    df = _load_csv(csv_source)
    df["date"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
    df = df.sort_values("date").reset_index(drop=True)

    start_date = df["date"].min()
    prices = df["Prices"].values

    def _days(d):
        return (pd.Timestamp(d) - start_date).days

    time = np.array([_days(d) for d in df["date"]])

    # OLS linear trend
    xbar, ybar = time.mean(), prices.mean()
    slope = np.sum((time - xbar) * (prices - ybar)) / np.sum((time - xbar) ** 2)
    intercept = ybar - slope * xbar

    # Sinusoidal seasonal (bilinear projection, period = 365 days)
    residuals = prices - (slope * time + intercept)
    sin_t = np.sin(time * 2 * np.pi / 365)
    cos_t = np.cos(time * 2 * np.pi / 365)
    u = np.sum(residuals * sin_t) / np.sum(sin_t ** 2)
    w = np.sum(residuals * cos_t) / np.sum(cos_t ** 2)
    amplitude = np.sqrt(u ** 2 + w ** 2)
    phase = np.arctan2(w, u)

    # Exact lookup for observed month-end dates
    exact = {_days(d): p for d, p in zip(df["date"], prices)}

    def estimate_price(input_date) -> float:
        d = _days(input_date)
        if d in exact:
            return round(float(exact[d]), 4)
        p = amplitude * np.sin(d * 2 * np.pi / 365 + phase) + slope * d + intercept
        return round(float(p), 4)

    return estimate_price

# 2.  GENERALISED CONTRACT PRICER
def price_storage_contract(
    injection_dates: list,
    withdrawal_dates: list,
    injection_volumes: list,
    withdrawal_volumes: list,
    max_injection_rate: float,
    max_withdrawal_rate: float,
    max_volume: float,
    monthly_storage_cost: float,
    csv_path: str  = None,
    verbose: bool = True,
) -> dict:
    """
    Price a generalised natural gas storage contract with multiple legs.

    Parameters: 
        injection_dates : list of str/date  — dates to buy and inject gas
        withdrawal_dates : list of str/date  — dates to withdraw and sell gas
        injection_volumes : list of float     — MMBtu to inject on each date
        withdrawal_volumes : list of float     — MMBtu to withdraw on each date
        max_injection_rate : float  — MMBtu/day ceiling on injections
        max_withdrawal_rate : float  — MMBtu/day ceiling on withdrawals
        max_volume : float  — tank capacity in MMBtu
        monthly_storage_cost : float  — flat $/month facility rental fee
        csv_path : str or None
            Path to Nat_Gas.csv.
            • None (default) → auto-detect: script folder → cwd → GitHub URL
            - Explicit path  -> e.g. csv_path=r"C:/Users/kokhe/.../Nat_Gas.csv"
        verbose : bool   — print blotter and P&L if True

    Returns:
        dict with keys:
            cash_in, cash_out, storage_cost, net_contract_value,
            profitable (bool), legs (list), warnings (list)
    """
    # Validate inputs 
    if len(injection_dates) != len(injection_volumes):
        raise ValueError("injection_dates and injection_volumes must be the same length.")
    if len(withdrawal_dates) != len(withdrawal_volumes):
        raise ValueError("withdrawal_dates and withdrawal_volumes must be the same length.")
    if not injection_dates and not withdrawal_dates:
        raise ValueError("Contract must have at least one injection or withdrawal date.")
    if max_volume <= 0:
        raise ValueError("max_volume must be positive.")
    if max_injection_rate <= 0 or max_withdrawal_rate <= 0:
        raise ValueError("Rate parameters must be positive.")

    source = _resolve_csv(csv_path)
    estimate_price = _build_price_model(source)

    # Build unified sorted event list
    events = []
    for d, v in zip(injection_dates, injection_volumes):
        events.append({"date": pd.Timestamp(d), "type": "inject",   "requested": float(v)})
    for d, v in zip(withdrawal_dates, withdrawal_volumes):
        events.append({"date": pd.Timestamp(d), "type": "withdraw", "requested": float(v)})

    # Chronological order; injections before withdrawals on the same day
    events.sort(key=lambda e: (e["date"], 0 if e["type"] == "inject" else 1))

    # Walk through events 
    cash_in = 0.0
    cash_out = 0.0
    storage_cost = 0.0
    current_vol = 0.0
    prev_date = None
    blotter = []
    contract_warnings = []

    daily_storage_rate = monthly_storage_cost / 30.0   # flat fee, not per-MMBtu

    for ev in events:
        ev_date = ev["date"]
        ev_type = ev["type"]
        requested = ev["requested"]
        price = estimate_price(ev_date)

        # Accrue flat storage fee for the gap since the previous event
        if prev_date is not None:
            days_held = (ev_date - prev_date).days
            storage_cost += daily_storage_rate * days_held

        if ev_type == "inject":
            after_rate = min(requested, max_injection_rate)
            space_left = max_volume - current_vol
            actual = max(min(after_rate, space_left), 0.0)

            if actual < requested:
                contract_warnings.append(
                    f"INJECT {ev_date.date()}: requested {requested:,.0f} MMBtu; "
                    f"only {actual:,.0f} injected "
                    f"(rate cap: {max_injection_rate:,.0f}, "
                    f"space remaining: {space_left:,.0f})."
                )

            cash_out += actual * price
            current_vol += actual
            blotter.append({
                "date": ev_date.date(), "type": "INJECT",
                "requested": requested, "actual": actual,
                "price": price, "cashflow": -actual * price,
                "volume_after": current_vol,
            })

        else:  # withdraw
            after_rate = min(requested, max_withdrawal_rate)
            actual = max(min(after_rate, current_vol), 0.0)

            if actual < requested:
                contract_warnings.append(
                    f"WITHDRAW {ev_date.date()}: requested {requested:,.0f} MMBtu; "
                    f"only {actual:,.0f} withdrawn "
                    f"(rate cap: {max_withdrawal_rate:,.0f}, "
                    f"in tank: {current_vol:,.0f})."
                )

            cash_in += actual * price
            current_vol -= actual
            blotter.append({
                "date": ev_date.date(), "type": "WITHDRAW",
                "requested": requested, "actual": actual,
                "price": price, "cashflow": actual * price,
                "volume_after": current_vol,
            })

        prev_date = ev_date

    net_contract_value = cash_in - cash_out - storage_cost

    # Verbose blotter
    if verbose:
        W = 72
        print("\n" + "=" * W)
        print("  NATURAL GAS STORAGE CONTRACT — VALUATION BLOTTER")
        print("=" * W)
        print(f"  {'Date':<13} {'Type':<10} {'Req (MMBtu)':>12} {'Act (MMBtu)':>12} "
              f"{'Price':>7} {'Cash Flow':>16} {'Vol After':>12}")
        print("  " + "-" * (W - 2))
        for leg in blotter:
            cf     = leg["cashflow"]
            cf_str = ("+" if cf >= 0 else "") + f"${cf:>15,.2f}"
            print(f"  {str(leg['date']):<13} {leg['type']:<10} "
                  f"{leg['requested']:>12,.0f} {leg['actual']:>12,.0f} "
                  f"{leg['price']:>7.2f} {cf_str:>16} {leg['volume_after']:>12,.0f}")
        print("  " + "-" * (W - 2))
        print(f"  {'Revenue (gas sales):':<44} ${cash_in:>15,.2f}")
        print(f"  {'Cost    (gas purchases):':<44} ${cash_out:>15,.2f}")
        print(f"  {'Storage cost (flat monthly fee x days/30):':<44} ${storage_cost:>15,.2f}")
        print("  " + "-" * (W - 2))
        verdict = "PROFITABLE" if net_contract_value > 0 else "LOSS"
        print(f"  {'Net contract value:':<44} ${net_contract_value:>15,.2f}   [{verdict}]")
        print("=" * W)
        if contract_warnings:
            print("\n  Constraint warnings:")
            for w in contract_warnings:
                print(f"    * {w}")
        print()

    return {
        "cash_in": round(cash_in, 2),
        "cash_out": round(cash_out, 2),
        "storage_cost": round(storage_cost, 2),
        "net_contract_value": round(net_contract_value, 2),
        "profitable": net_contract_value > 0,
        "legs": blotter,
        "warnings": contract_warnings,
    }


# 3.  TEST CASES
if __name__ == "__main__":

    # csv_path=None  ->  auto-detects Nat_Gas.csv in the same folder as this script
    CSV = None

    # TEST 1 — Single buy / single sell, no constraints bind
    print("\n" + "-" * 72)
    print("  TEST 1 — Single buy / single sell, no constraints bind")
    print("-" * 72)
    price_storage_contract(
        injection_dates      = ["2024-06-30"],
        withdrawal_dates     = ["2025-01-31"],
        injection_volumes    = [1_000_000],
        withdrawal_volumes   = [1_000_000],
        max_injection_rate   = 5_000_000,
        max_withdrawal_rate  = 5_000_000,
        max_volume           = 5_000_000,
        monthly_storage_cost = 100_000,
        csv_path             = CSV,
    )

    # TEST 2 — Two injections, two withdrawals (staggered legs)
    print("-" * 72)
    print("  TEST 2 — Two injections / two withdrawals (staggered legs)")
    print("-" * 72)
    price_storage_contract(
        injection_dates      = ["2024-05-31", "2024-06-30"],
        withdrawal_dates     = ["2024-12-31", "2025-01-31"],
        injection_volumes    = [500_000, 500_000],
        withdrawal_volumes   = [500_000, 500_000],
        max_injection_rate   = 5_000_000,
        max_withdrawal_rate  = 5_000_000,
        max_volume           = 2_000_000,
        monthly_storage_cost = 80_000,
        csv_path             = CSV,
    )

    # TEST 3 — Injection rate constraint binds
    print("-" * 72)
    print("  TEST 3 — Injection rate constraint binds (500K/day cap)")
    print("-" * 72)
    price_storage_contract(
        injection_dates      = ["2024-06-30"],
        withdrawal_dates     = ["2025-01-31"],
        injection_volumes    = [2_000_000],
        withdrawal_volumes   = [2_000_000],
        max_injection_rate   = 500_000,
        max_withdrawal_rate  = 5_000_000,
        max_volume           = 5_000_000,
        monthly_storage_cost = 50_000,
        csv_path             = CSV,
    )

    # TEST 4 — Tank capacity binds on second injection
    print("-" * 72)
    print("  TEST 4 — Tank capacity binds on second injection (max 1M MMBtu)")
    print("-" * 72)
    price_storage_contract(
        injection_dates      = ["2024-05-31", "2024-06-30"],
        withdrawal_dates     = ["2025-01-31"],
        injection_volumes    = [800_000, 800_000],
        withdrawal_volumes   = [1_000_000],
        max_injection_rate   = 5_000_000,
        max_withdrawal_rate  = 5_000_000,
        max_volume           = 1_000_000,
        monthly_storage_cost = 120_000,
        csv_path             = CSV,
    )

    # TEST 5 — Unprofitable: storage fee swamps the spread
    print("-" * 72)
    print("  TEST 5 — Unprofitable: storage fee swamps the spread")
    print("-" * 72)
    price_storage_contract(
        injection_dates      = ["2024-05-31"],
        withdrawal_dates     = ["2025-01-31"],
        injection_volumes    = [50_000],
        withdrawal_volumes   = [50_000],
        max_injection_rate   = 5_000_000,
        max_withdrawal_rate  = 5_000_000,
        max_volume           = 5_000_000,
        monthly_storage_cost = 500_000,
        csv_path             = CSV,
    )