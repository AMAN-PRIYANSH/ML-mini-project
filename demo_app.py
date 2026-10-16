"""Streamlit demo for the Lending Club mini-project.
Run locally after downloading demo_artifacts/ from the Colab notebook:
    pip install streamlit xgboost pandas numpy scikit-learn
    streamlit run demo_app.py
Opens on http://localhost:8501
"""
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Lending Club — Honest Risk Model", layout="wide")
st.title("Lending Club default risk — honest model")
st.caption("Tuned XGBoost, trained on 2012-2014 loans WITHOUT the leaky credit-score columns, tested on 2015 loans.")

@st.cache_resource
def load_artifacts():
    pp = pickle.load(open("demo_artifacts/preproc.pkl", "rb"))
    tc = pickle.load(open("demo_artifacts/test_cases.pkl", "rb"))
    model = xgb.XGBClassifier()
    model.load_model("demo_artifacts/xgb_tuned.json")
    return pp, tc, model

pp, tc, model = load_artifacts()
test_df = tc["test_full"]
tested_rows = tc["rows"]

def to_vector(row):
    """Turn one loan row into the model's input vector (same preprocessing as training)."""
    v = np.zeros(len(pp["feature_names"]), dtype=np.float32)
    for k, c in enumerate(pp["num_cols"]):
        x = row[c]
        if isinstance(x, str):
            x = x.replace("%", "")
            try: x = float(x) / 100.0
            except ValueError: x = np.nan
        if pd.isna(x): x = pp["median"][c]
        v[k] = x
    j = len(pp["num_cols"])
    for c in pp["cat_cols"]:
        values = pp["cat_values"][c]
        val = str(row[c]) if pd.notna(row[c]) else "missing"
        if val in values:
            v[j + values.index(val)] = 1.0
        j += len(values)
    v = (v - pp["mean"]) / pp["std"]
    return np.nan_to_num(v)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Pick a 2015 loan")
    row_idx = st.number_input(f"Loan number (0 to {len(test_df)-1:,})",
                              min_value=0, max_value=len(test_df)-1,
                              value=int(tested_rows[0]), step=1)
    st.caption("Pre-tested loan numbers that give clean results:")
    st.code(", ".join(str(r) for r in tested_rows), language=None)

row = test_df.iloc[int(row_idx)]
x = to_vector(row).reshape(1, -1)
prob_paid = float(model.predict_proba(x)[0, 1])
prob_default = 1 - prob_paid
verdict = "HIGH risk of default" if prob_default >= 0.5 else "LOW risk of default"
truth = "Defaulted / Charged Off" if row["loan_status"] != "Fully Paid" else "Fully Paid"

with col2:
    st.subheader("Model says")
    st.metric("Predicted chance of default", f"{prob_default*100:.1f}%", verdict)
    st.subheader("What really happened")
    st.write(f"**{truth}**")
    match = (prob_default >= 0.5) == (row["loan_status"] != "Fully Paid")
    if match:
        st.success("Model's call matches reality for this loan.")
    else:
        st.warning("Model's call does NOT match for this loan (honest model, ROC-AUC ~0.74).")

st.divider()
st.subheader("Loan application details")
show_cols = ["loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
             "annual_inc", "dti", "fico_range_low", "fico_range_high",
             "emp_length", "home_ownership", "purpose", "addr_state"]
st.table(pd.DataFrame({"Feature": show_cols, "Value": [str(row[c]) for c in show_cols]}))
