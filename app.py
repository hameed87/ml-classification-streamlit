"""
Streamlit app entry point

Author: Abdul
Date: February 10, 2026
Description: Main entry point for ML Classification Streamlit application
"""

import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

TARGET = "default.payment.next.month"
ARTIFACT_DIR = os.path.join("model", "artifacts")

MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl",
}

st.set_page_config(page_title="ML Classification App", layout="wide")
st.title("ML Classification App")
st.write("Upload test data and select a model to view evaluation metrics, confusion matrix and report.")

@st.cache_resource
def load_scaler():
    with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model(model_filename: str):
    with open(os.path.join(ARTIFACT_DIR, model_filename), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metrics():
    path = os.path.join(ARTIFACT_DIR, "metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

left, right = st.columns([1, 1])

with left:
    selected_model = st.selectbox("Select Model", list(MODEL_FILES.keys()))
    st.subheader("Saved Evaluation Metrics (from training)")
    metrics_df = load_metrics()
    if metrics_df is None:
        st.warning("metrics.csv not found. First run: python -m model.train_and_save")
    else:
        st.dataframe(metrics_df, use_container_width=True)

with right:
    uploaded = st.file_uploader("Upload Test CSV (small file recommended)", type=["csv"])
    st.info("Tip: Streamlit free tier has limited capacity. Upload only test data.")

if uploaded:
    df = pd.read_csv(uploaded)

    # Drop ID if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    scaler = load_scaler()
    model = load_model(MODEL_FILES[selected_model])

    if TARGET in df.columns:
        X = df.drop(columns=[TARGET])
        y = df[TARGET].astype(int)

        Xs = scaler.transform(X)
        pred = model.predict(Xs)

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y, pred))

        st.subheader("Classification Report")
        st.text(classification_report(y, pred, digits=4))
    else:
        st.warning(f"Target column '{TARGET}' not found. Showing predictions only.")
        Xs = scaler.transform(df)
        pred = model.predict(Xs)

        out = df.copy()
        out["prediction_default_next_month"] = pred
        st.subheader("Predictions Preview")
        st.dataframe(out.head(30), use_container_width=True)


