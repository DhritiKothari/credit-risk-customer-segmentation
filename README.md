# Credit Risk Analysis × Customer Segmentation

An end-to-end machine-learning project combining **credit-risk classification** and **customer segmentation** for banking decision support.

## Project Overview

The project contains two complementary ML pipelines:

1. **Credit Risk Analysis** — classifies a credit applicant as **Good (Low Risk)** or **Bad (High Risk)** using the German Credit Risk dataset.
2. **Customer Segmentation** — groups customers into six behavioural segments using **K-Means clustering** and predicts the segment of a new customer.

The project includes exploratory data analysis, feature engineering, model development, evaluation, serialized models, and Streamlit applications.

## System Architecture

```text
Raw Data
   ↓
Data Cleaning & EDA
   ↓
Feature Engineering
   ↓
Encoding / Scaling
   ↓
┌───────────────────────────┬────────────────────────────┐
│ Credit Risk Classification │ Customer Segmentation      │
│ DT / RF / Extra Trees      │ K-Means (K = 6)            │
│ / XGBoost                  │ PCA + cluster validation   │
└──────────────┬────────────┴──────────────┬─────────────┘
               ↓                           ↓
        Risk Prediction              Segment Profile
               └──────────────┬────────────┘
                              ↓
                    Decision Support
                              ↓
                         Streamlit
```

## Credit Risk Pipeline

- Dataset: German Credit Risk
- Target: `Good` / `Bad`
- Important features: Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration
- Candidate models: Decision Tree, Random Forest, Extra Trees, XGBoost
- Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix and ROC-AUC where applicable
- Saved deployment model: `models/extra_trees_credit_model.pkl`

## Customer Segmentation Pipeline

- Dataset: Customer Personality Analysis
- Feature groups: demographic, financial, purchase behaviour and engagement
- Engineered features include Age, Total Spending and purchase/engagement measures
- Standardization is applied before K-Means
- K selected using the Elbow Method; final serialized model uses **K = 6**
- PCA is used to visualize the resulting clusters
- Saved models: `models/kmeans_model.pkl` and `models/scaler.pkl`

## Run the Streamlit Apps

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run credit-risk prediction:

```bash
streamlit run apps/credit_risk_app.py
```

Run customer segmentation:

```bash
streamlit run apps/customer_segmentation_app.py
```

## Repository Structure

```text
credit-risk-customer-segmentation/
├── apps/
│   ├── credit_risk_app.py
│   └── customer_segmentation_app.py
├── data/
│   ├── german_credit_data.csv
│   └── customer_segmentation.csv
├── models/
│   ├── extra_trees_credit_model.pkl
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── *_encoder.pkl
├── notebooks/
│   ├── Credit_Risk_Scoring.ipynb
│   └── Customer_Segmentation_Analysis.ipynb
├── visualizations/
├── docs/
│   ├── Credit_Risk_Modeling_Report.docx
│   ├── Customer_Segmentation_Report.pdf
│   └── A3_Conference_Poster.*
├── requirements.txt
├── .gitignore
└── README.md
```

## Notes

The serialized `.pkl` files are included so the Streamlit applications can run without retraining. For reproducible production deployment, keep the training environment/version aligned with the environment used to serialize the models.

This repository is an academic/project implementation and should not be used as a standalone real-world lending decision system without validation, explainability, fairness testing, privacy/security controls, regulatory review, monitoring and retraining.
