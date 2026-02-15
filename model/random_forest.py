"""
Random Forest model code

Author: Abdul
Date: February 10, 2026
Description: Random Forest classification model implementation
"""
from sklearn.ensemble import RandomForestClassifier

def build_model():
    return RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=7,
        n_jobs=-1
    )
