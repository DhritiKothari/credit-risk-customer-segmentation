from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="centered")
st.title("Customer Segmentation App")
st.write("Enter customer details to predict the K-Means segment.")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=1_000_000, value=50_000)
total_spending = st.number_input("Total Spending (sum of purchases)", min_value=0, max_value=10_000, value=500)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=5)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=5)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=100, value=10)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
    "Income": [income],
    "Age": [age],
    "Recency": [recency],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
})
input_data = input_data[list(scaler.feature_names_in_)]
input_data_scaled = scaler.transform(input_data)

segment_descriptions = {
    0: "Older / lower-spend profile",
    1: "Premium / high-value profile",
    2: "Engaged / digital-oriented profile",
    3: "High-value / potentially at-risk profile",
    4: "Budget / low-spend profile",
    5: "Low-engagement / potentially at-risk profile",
}

if st.button("Predict Segment", type="primary"):
    cluster = int(kmeans.predict(input_data_scaled)[0])
    st.success(f"Predicted customer segment: **{cluster}**")
    st.write(f"**Profile:** {segment_descriptions.get(cluster, 'Profile unavailable')}")
else:
    st.info("Enter customer details and click **Predict Segment**.")
