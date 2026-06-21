"""
loan_default_model.py
═════════════════════
Probability of Default (PD) Model & Expected Loss Calculator
JPMorgan Chase Quantitative Research — Task 3

Background
----------
The retail bank has observed higher-than-expected default rates on personal
loans.  This module trains a Logistic Regression model on historical borrower
data to estimate PD, then computes the Expected Loss on any loan:

    Expected Loss = PD × LGD × EAD

where:
    PD  = Probability of Default  (predicted by this model)
    LGD = Loss Given Default       = 1 − Recovery Rate = 1 − 0.10 = 0.90
    EAD = Exposure at Default      = loan_amt_outstanding

Model choice: Logistic Regression
----------------------------------
Logistic Regression is the industry-standard starting point for PD modelling
(Basel II/III internal ratings-based approach).  It:
  • outputs a well-calibrated probability in [0, 1]
  • is fully interpretable via signed coefficients
  • satisfies regulatory transparency requirements
  • achieves AUC ≈ 1.0 on this dataset (strong feature signal from
    credit_lines_outstanding and debt-to-income metrics)

A Gradient Boosting model is also trained for comparison.

Data source
-----------
CSV is loaded from the local file by default, with a GitHub URL fallback.
"""

import os
import io
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                             brier_score_loss, log_loss)

# Data source
_GITHUB_URL = (
    "https://raw.githubusercontent.com/"
    "kokheetankh-gif/Investigate-and-analyze-price-data/main/"
    "Task%203%20and%204_Loan_Data.csv"
)

FEATURES = [
    'credit_lines_outstanding',   # strong positive predictor (more lines → higher risk)
    'loan_amt_outstanding',       # loan size
    'total_debt_outstanding',     # total debt burden
    'income',                     # earnings capacity (negative predictor)
    'years_employed',             # employment stability (negative predictor)
    'fico_score',                 # credit score (negative predictor)
]
TARGET  = 'default'
RECOVERY_RATE   = 0.10    # 10% as stated in the brief
LGD  = 1 - RECOVERY_RATE   # Loss Given Default = 90%


# Data loading 
def _load_data(csv_path=None) -> pd.DataFrame:
    """Load loan data from a local CSV or the GitHub URL."""
    # Priority: explicit arg → script dir → cwd → GitHub
    if csv_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [
            os.path.join(script_dir, "Task_3_and_4_Loan_Data.csv"),
            os.path.join(os.getcwd(),  "Task_3_and_4_Loan_Data.csv"),
        ]:
            if os.path.exists(candidate):
                csv_path = candidate
                break

    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    # Fallback: fetch from GitHub
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


# ── Model training ─────────────────────────────────────────────────────────────
def train_model(csv_path=None, verbose=True):
    """
    Train the PD model on the loan dataset.

    Returns a dict containing:
        model     : fitted LogisticRegression
        scaler    : fitted StandardScaler
        features  : list of feature names
        test_auc  : AUC-ROC on held-out test set
        X_test, y_test, y_proba : test data and predictions (for diagnostics)
    """
    df = _load_data(csv_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_s, y_train)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    ll = log_loss(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    if verbose:
        W = 60
        print("\n" + "=" * W)
        print("  PD MODEL — LOGISTIC REGRESSION — TRAINING SUMMARY")
        print("=" * W)
        print(f"  Training samples  : {len(X_train):,}")
        print(f"  Test samples      : {len(X_test):,}")
        print(f"  Base default rate : {y.mean():.1%}")
        print(f"  AUC-ROC           : {auc:.4f}")
        print(f"  Log-Loss          : {ll:.4f}")
        print(f"  Brier Score       : {brier:.4f}")
        print()
        print("  Feature coefficients (scaled):")
        for feat, coef in sorted(
            zip(FEATURES, model.coef_[0]), key=lambda x: -abs(x[1])
        ):
            direction = "↑ risk" if coef > 0 else "↓ risk"
            print(f"    {feat:<30}: {coef:+.4f}  ({direction})")
        print(f"    {'Intercept':<30}: {model.intercept_[0]:+.4f}")
        print()
        print("  Classification report (threshold = 0.5):")
        pred = (y_proba >= 0.5).astype(int)
        print(classification_report(y_test, pred, target_names=["No Default", "Default"],
                                    digits=3))
        print("=" * W)

    return {
        'model':   model,
        'scaler':  scaler,
        'features': FEATURES,
        'test_auc': auc,
        'X_test':  X_test,
        'y_test':  y_test,
        'y_proba': y_proba,
    }


# Core PD function
def predict_pd(
    credit_lines_outstanding: float,
    loan_amt_outstanding: float,
    total_debt_outstanding: float,
    income: float,
    years_employed: float,
    fico_score: float,
    model_bundle: dict,
) -> float:
    """
    Predict the Probability of Default (PD) for a single borrower.

    Parameters:
        credit_lines_outstanding : int/float — number of active credit lines
        loan_amt_outstanding     : float     — current outstanding loan balance ($)
        total_debt_outstanding   : float     — total debt across all obligations ($)
        income                   : float     — annual income ($)
        years_employed           : int/float — years in current employment
        fico_score               : int/float — FICO credit score (300–850)
        model_bundle             : dict      — output of train_model()

    Returns:
        float : Probability of Default in [0, 1]
    """
    model  = model_bundle['model']
    scaler = model_bundle['scaler']

    row = pd.DataFrame([{
        'credit_lines_outstanding': credit_lines_outstanding,
        'loan_amt_outstanding': loan_amt_outstanding,
        'total_debt_outstanding': total_debt_outstanding,
        'income': income,
        'years_employed': years_employed,
        'fico_score': fico_score,
    }])[FEATURES]

    row_scaled = scaler.transform(row)
    pd_val = model.predict_proba(row_scaled)[0, 1]
    return round(float(pd_val), 6)


# Expected Loss function
def expected_loss(
    credit_lines_outstanding: float,
    loan_amt_outstanding: float,
    total_debt_outstanding: float,
    income: float,
    years_employed: float,
    fico_score: float,
    model_bundle: dict,
    recovery_rate: float = RECOVERY_RATE,
    verbose: bool  = True,
) -> dict:
    """
    Compute the Expected Loss (EL) on a loan given borrower characteristics.

    Formula:
        EL = PD × LGD × EAD
        LGD = 1 − recovery_rate   (Loss Given Default)
        EAD = loan_amt_outstanding (Exposure at Default)

    Parameters:
        (same borrower parameters as predict_pd, plus:)
        recovery_rate : float — fraction recovered in default event (default: 0.10)
        verbose : bool  — if True, print a breakdown of the calculation

    Returns:
        dict with keys:
            pd : float — probability of default
            lgd : float — loss given default (= 1 − recovery_rate)
            ead : float — exposure at default (loan amount)
            expected_loss : float — EL in dollars
            risk_category : str   — Low / Medium / High / Very High
    """
    pd_val = predict_pd(
        credit_lines_outstanding, loan_amt_outstanding,
        total_debt_outstanding, income, years_employed,
        fico_score, model_bundle
    )

    lgd_val = 1.0 - recovery_rate
    ead_val = loan_amt_outstanding
    el_val = pd_val * lgd_val * ead_val

    # Risk banding
    if pd_val < 0.05:
        risk_cat = "Low"
    elif pd_val < 0.20:
        risk_cat = "Medium"
    elif pd_val < 0.60:
        risk_cat = "High"
    else:
        risk_cat = "Very High"

    if verbose:
        print(f"\n  {'─'*44}")
        print(f"  Borrower Assessment")
        print(f"  {'─'*44}")
        print(f"  Credit lines outstanding : {credit_lines_outstanding}")
        print(f"  Loan amount outstanding  : ${loan_amt_outstanding:>12,.2f}")
        print(f"  Total debt outstanding   : ${total_debt_outstanding:>12,.2f}")
        print(f"  Income                   : ${income:>12,.2f}")
        print(f"  Years employed           : {years_employed}")
        print(f"  FICO score               : {fico_score}")
        print(f"  {'─'*44}")
        print(f"  PD  (Probability of Default)  : {pd_val:.4%}")
        print(f"  LGD (Loss Given Default)       : {lgd_val:.0%}  (recovery = {recovery_rate:.0%})")
        print(f"  EAD (Exposure at Default)      : ${ead_val:>12,.2f}")
        print(f"  {'─'*44}")
        print(f"  Expected Loss (EL = PD×LGD×EAD): ${el_val:>12,.2f}")
        print(f"  Risk category                  : {risk_cat}")
        print(f"  {'─'*44}\n")

    return {
        'pd':             pd_val,
        'lgd':            lgd_val,
        'ead':            ead_val,
        'expected_loss':  round(el_val, 2),
        'risk_category':  risk_cat,
    }


# ── Model comparison (bonus) ───────────────────────────────────────────────────
def compare_models(csv_path=None):
    """
    Train and compare Logistic Regression vs Gradient Boosting.
    Prints a side-by-side metric table.
    """
    df = _load_data(csv_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    candidates = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            True   # needs scaled features
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, random_state=42),
            False
        ),
    }

    print(f"\n{'Model':<22} {'AUC-ROC':>9} {'Log-Loss':>10} {'Brier':>8}")
    print("─" * 54)
    for name, (m, scaled) in candidates.items():
        Xtr = X_train_s if scaled else X_train
        Xte = X_test_s  if scaled else X_test
        m.fit(Xtr, y_train)
        p   = m.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_test, p)
        ll  = log_loss(y_test, p)
        br  = brier_score_loss(y_test, p)
        print(f"  {name:<20} {auc:>9.4f} {ll:>10.4f} {br:>8.4f}")
    print()


# Demo 
if __name__ == "__main__":

    CSV = None   # auto-detect from script folder / cwd / GitHub

    # Step 1: train
    bundle = train_model(csv_path=CSV, verbose=True)

    # Step 2: model comparison
    print("\n" + "=" * 54)
    print("  MODEL COMPARISON: Logistic Regression vs Gradient Boosting")
    print("=" * 54)
    compare_models(csv_path=CSV)

    # Step 3: example loans
    print("=" * 54)
    print("  SAMPLE LOAN ASSESSMENTS")
    print("=" * 54)

    test_cases = [
        # (label, credit_lines, loan_amt, total_debt, income, yrs_employed, fico)
        ("Low-risk borrower",
         0, 3000, 2000, 75000, 5, 720),
        ("Medium-risk borrower",
         2, 4500, 6000, 45000, 3, 620),
        ("High-risk borrower",
         4, 5000, 12000, 35000, 2, 570),
        ("Very high-risk borrower",
         5, 6000, 15000, 28000, 1, 510),
    ]

    summary_rows = []
    for label, cl, la, td, inc, ye, fico in test_cases:
        print(f"\n  [{label}]")
        res = expected_loss(
            credit_lines_outstanding = cl,
            loan_amt_outstanding     = la,
            total_debt_outstanding   = td,
            income                   = inc,
            years_employed           = ye,
            fico_score               = fico,
            model_bundle             = bundle,
        )
        summary_rows.append((label, res['pd'], res['expected_loss'], res['risk_category']))

    print("=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Borrower':<26} {'PD':>8} {'Exp. Loss':>12} {'Risk':>12}")
    print("  " + "─" * 56)
    for label, pd_val, el, risk in summary_rows:
        print(f"  {label:<26} {pd_val:>8.2%} {el:>12,.2f}   {risk}")
    print("=" * 60)