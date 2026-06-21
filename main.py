import sys
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, date
warnings.filterwarnings("ignore")

_DATA_PATH = os.path.join(os.path.dirname(__file__), "Nat_Gas.csv")
if not os.path.exists(_DATA_PATH):
    # Fallback to the upload location used in the analysis environment
    _DATA_PATH = "/mnt/user-data/uploads/Nat_Gas.csv"

# Load and fit the model once at import time
def _build_model(csv_path: str):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month

    t0 = df["date"].min()  # reference: 31 Oct 2020

    def _months(d):
        """Numeric offset in months from t0."""
        ts = pd.Timestamp(d)
        return (ts.year - t0.year) * 12 + (ts.month - t0.month)

    df["t"] = df["date"].apply(_months)
    trend_coef = np.polyfit(df["t"], df["Prices"], 1)
    trend_fn = np.poly1d(trend_coef)
    df["trend"] = trend_fn(df["t"])
    df["residual"] = df["Prices"] - df["trend"]
    seasonal_adj = df.groupby("month")["residual"].mean()
    resid_std = df["residual"].std()
    last_date = df["date"].max()

    return t0, trend_fn, seasonal_adj, resid_std, last_date

_T0, _TREND_FN, _SEASONAL_ADJ, _RESID_STD, _LAST_DATE = _build_model(_DATA_PATH)


def estimate_price(input_date, return_bounds: bool = False):
    """
    Estimate the natural gas purchase price for any calendar date.

    Parameters:
        input_date : str | datetime | date | pd.Timestamp
            Target date.  Common string formats are accepted:
              '2025-06-30'  |  '6/30/2025'  |  '30 Jun 2025'  |  '2025-06'
            Only the year and month are used; day-of-month is ignored.

        return_bounds : bool, default False
            If True, also return a ±2σ uncertainty interval.

    Returns:
        float
            Estimated price ($/MMBtu), rounded to 2 dp.
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
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    print("=" * 65)
    print("  Natural Gas Price Estimator")
    print("  Data window : Oct 2020 – Sep 2024")
    print("  Model       : Linear trend + calendar-month seasonal adjustment")
    print("=" * 65)

    print("\n── Historical spot-checks (observed vs estimated) ──")
    spot_checks = [
        ("2021-03-31", 10.90),
        ("2022-06-30", 10.40),
        ("2023-01-31", 12.10),
        ("2024-06-30", 11.50),
    ]
    print(f"  {'Date':<13}  {'Observed':>10}  {'Estimated':>10}  {'Error':>8}")
    print("  " + "─" * 47)
    for d, obs in spot_checks:
        est = estimate_price(d)
        print(f"  {d:<13}  {obs:>10.2f}  {est:>10.2f}  {est-obs:>+8.2f}")

    print("\n── 12-Month Extrapolation (Oct 2024 – Sep 2025) ──")
    print(f"  {'Month':<10}  {'Estimate':>10}  {'Lower(−2σ)':>12}  {'Upper(+2σ)':>12}")
    print("  " + "─" * 50)
    from dateutil.relativedelta import relativedelta
    for i in range(1, 13):
        d = _LAST_DATE + relativedelta(months=i)
        p, lo, hi = estimate_price(d, return_bounds=True)
        print(f"  {d.strftime('%b %Y'):<10}  {p:>10.2f}  {lo:>12.2f}  {hi:>12.2f}")

    print(f"\n  Residual σ = {_RESID_STD:.4f}  →  ±2σ band ≈ ±{2*_RESID_STD:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single date supplied on command line
        raw = sys.argv[1]
        try:
            p, lo, hi = estimate_price(raw, return_bounds=True)
            print(f"\nDate     : {pd.Timestamp(raw).strftime('%d %b %Y')}")
            print(f"Estimate : ${p:.2f} /MMBtu")
            print(f"Range    : ${lo:.2f} – ${hi:.2f}  (±2σ interval)")
        except Exception as e:
            print(f"Error parsing date '{raw}': {e}")
            sys.exit(1)
    else:
        _demo()