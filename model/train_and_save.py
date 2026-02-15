"""
Train and save machine learning models

Author: Abdul
Date: February 15, 2026
Description: Script to train all classification models and save them for deployment
"""

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from model.logistic_regression import build_model as lr_model
from model.decision_tree import build_model as dt_model
from model.knn import build_model as knn_model
from model.naive_bayes import build_model as nb_model
from model.random_forest import build_model as rf_model
from model.xgboost import build_model as xgb_model

DATA_PATH = os.path.join("data", "credit_default.csv")
TARGET = "default.payment.next.month"
ARTIFACT_DIR = os.path.join("model", "artifacts")

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Drop ID column
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Split X/y
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Train-test split (unique config to avoid identical outputs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    # Scaling (helps LR/KNN)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": lr_model(),
        "Decision Tree": dt_model(),
        "KNN": knn_model(),
        "Naive Bayes": nb_model(),
        "Random Forest": rf_model(),
        "XGBoost": xgb_model(),
    }

    rows = []

    for name, model in models.items():
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        rows.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred)
        })

        with open(os.path.join(ARTIFACT_DIR, f"{name.replace(' ', '_')}.pkl"), "wb") as f:
            pickle.dump(model, f)

    with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(ARTIFACT_DIR, "metrics.csv"), index=False)

    print("\n✅ Training complete. Artifacts saved in:", ARTIFACT_DIR)
    print("\n=== Metrics Comparison Table ===")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()
