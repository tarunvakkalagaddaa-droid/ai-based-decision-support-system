# AI-Based Decision Support System

A simple base project that helps a loan officer decide whether to approve a loan.

## How it works
1. **Data**: synthetic applicant data is generated (swap in your own CSV later).
2. **AI model**: a Random Forest predicts the probability of approval.
3. **Business rules**: checks such as credit score and debt-to-income ratio raise red flags.
4. **Decision**: combines both into APPROVE / REVIEW / REJECT, with the reasons shown.

## Run
```
pip install -r requirements.txt
python dss.py --demo   # sample applicants
python dss.py          # enter your own applicant details
```

## Ideas to extend
- Load a real dataset (e.g. `pd.read_csv("loans.csv")`) in place of `generate_data()`.
- Add SHAP for per-applicant explanations.
- Wrap it in a Streamlit or Flask web interface.
- Adapt it to other domains (hiring, inventory, healthcare triage) by changing `FEATURES` and `rule_checks()`.
