"""
XGBoost model code

Author: Abdul
Date: February 10, 2026
Description: XGBoost classification model implementation
"""
from xgboost import XGBClassifier

def build_model():
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=7,
        eval_metric="logloss"
    )
