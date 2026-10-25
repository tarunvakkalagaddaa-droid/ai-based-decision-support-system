"""
AI-Based Decision Support System (base project)
------------------------------------------------
Use case : Loan approval support.
Approach : ML model (Random Forest) estimates approval probability,
           business rules add safety checks, and the system returns a
           recommendation (APPROVE / REVIEW / REJECT) with explanations.

Run:
    pip install -r requirements.txt
    python dss.py            # interactive mode
    python dss.py --demo     # run with sample applicants
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ["age", "income", "credit_score", "loan_amount", "existing_debt", "employment_years"]


# ----------------------------------------------------------------------
# 1. Data (synthetic, replace with a real CSV for a real project)
# ----------------------------------------------------------------------
def generate_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(21, 65, n),
        "income": rng.normal(60000, 20000, n).clip(15000, 200000).round(),
        "credit_score": rng.normal(660, 80, n).clip(300, 850).round(),
        "loan_amount": rng.normal(20000, 10000, n).clip(1000, 80000).round(),
        "existing_debt": rng.normal(10000, 7000, n).clip(0, 60000).round(),
        "employment_years": rng.integers(0, 35, n),
    })
    # Hidden rule that generates the "true" label (with noise)
    score = (
        0.02 * (df["credit_score"] - 600)
        + 0.00004 * df["income"]
        - 0.00008 * df["loan_amount"]
        - 0.00006 * df["existing_debt"]
        + 0.05 * df["employment_years"]
        + rng.normal(0, 0.8, n)
    )
    df["approved"] = (score > score.median()).astype(int)
    return df


# ----------------------------------------------------------------------
# 2. Model
# ----------------------------------------------------------------------
def train_model(df):
    X, y = df[FEATURES], df["approved"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Model accuracy on test data: {accuracy_score(y_test, preds):.2%}")
    print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))
    return model


# ----------------------------------------------------------------------
# 3. Decision logic
# ----------------------------------------------------------------------
def rule_checks(a):
    """Hard business rules. Returns (list of red flags, list of positives)."""
    flags, positives = [], []
    dti = (a["existing_debt"] + a["loan_amount"]) / a["income"]
    if a["credit_score"] < 500:
        flags.append("Credit score below 500")
    elif a["credit_score"] >= 700:
        positives.append("Strong credit score (700+)")
    if dti > 0.8:
        flags.append(f"High debt-to-income ratio ({dti:.2f})")
    elif dti < 0.4:
        positives.append(f"Healthy debt-to-income ratio ({dti:.2f})")
    if a["loan_amount"] > 0.6 * a["income"] * 3:
        flags.append("Loan amount very large compared with income")
    if a["employment_years"] < 1:
        flags.append("Less than 1 year of employment")
    return flags, positives


def explain(model, applicant):
    """Simple explanation: top features by global importance with the applicant's value."""
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:3]
    return [f"{name} = {applicant[name]:,.0f} (model weight {w:.0%})" for name, w in imp]


def decide(model, applicant):
    X = pd.DataFrame([applicant])[FEATURES]
    prob = model.predict_proba(X)[0][1]
    flags, positives = rule_checks(applicant)

    if flags and prob < 0.6:
        decision = "REJECT"
    elif prob >= 0.75 and not flags:
        decision = "APPROVE"
    else:
        decision = "REVIEW (manual check recommended)"

    return {
        "decision": decision,
        "approval_probability": prob,
        "red_flags": flags,
        "positives": positives,
        "key_factors": explain(model, applicant),
    }


def show(result):
    print("\n" + "=" * 50)
    print(f"RECOMMENDATION : {result['decision']}")
    print(f"Approval probability : {result['approval_probability']:.1%}")
    if result["positives"]:
        print("Positives:")
        for p in result["positives"]:
            print(f"  + {p}")
    if result["red_flags"]:
        print("Red flags:")
        for f in result["red_flags"]:
            print(f"  ! {f}")
    print("Most influential factors:")
    for k in result["key_factors"]:
        print(f"  - {k}")
    print("=" * 50)


# ----------------------------------------------------------------------
# 4. Interface
# ----------------------------------------------------------------------
def ask(prompt, cast=float):
    while True:
        try:
            return cast(input(prompt))
        except ValueError:
            print("Please enter a valid number.")


def interactive(model):
    print("\nEnter applicant details (Ctrl+C to quit)")
    while True:
        applicant = {
            "age": ask("Age: "),
            "income": ask("Annual income: "),
            "credit_score": ask("Credit score (300-850): "),
            "loan_amount": ask("Loan amount requested: "),
            "existing_debt": ask("Existing debt: "),
            "employment_years": ask("Years of employment: "),
        }
        show(decide(model, applicant))
        if input("\nAnother applicant? (y/n): ").strip().lower() != "y":
            break


def demo(model):
    samples = [
        {"age": 35, "income": 90000, "credit_score": 760, "loan_amount": 15000, "existing_debt": 5000, "employment_years": 10},
        {"age": 28, "income": 40000, "credit_score": 590, "loan_amount": 30000, "existing_debt": 20000, "employment_years": 2},
        {"age": 45, "income": 55000, "credit_score": 650, "loan_amount": 25000, "existing_debt": 12000, "employment_years": 8},
    ]
    for s in samples:
        print("\nApplicant:", s)
        show(decide(model, s))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Based Decision Support System")
    parser.add_argument("--demo", action="store_true", help="run with sample applicants")
    args = parser.parse_args()

    data = generate_data()
    model = train_model(data)
    demo(model) if args.demo else interactive(model)
