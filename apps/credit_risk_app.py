from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "extra_trees_credit_model.pkl")
encoders = {
    "Sex": joblib.load(MODEL_DIR / "Sex_encoder.pkl"),
    "Housing": joblib.load(MODEL_DIR / "Housing_encoder.pkl"),
    "Saving accounts": joblib.load(MODEL_DIR / "Saving_accounts_encoder.pkl"),
    "Checking account": joblib.load(MODEL_DIR / "Checking_account_encoder.pkl"),
}

st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="centered")
st.title("Credit Risk Prediction App")
st.write("Enter applicant details to predict the credit-risk category.")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", options=["male", "female"])
job = st.number_input("Job (0–3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", options=["own", "rent", "free"])
saving_accounts = st.selectbox("Saving accounts", options=["little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking account", options=["little", "moderate", "rich"])
credit_amount = st.number_input("Credit amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration],
})

if st.button("Predict Risk", type="primary"):
    pred = model.predict(input_data)[0]
    if str(pred).lower() == "good":
        st.success("Predicted credit risk: **Good (Low Risk)**")
    else:
        st.error("Predicted credit risk: **Bad (High Risk)**")
