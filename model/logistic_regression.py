"""
Logistic Regression model code

Author: Abdul
Date: February 10, 2026
Description: Logistic Regression classification model implementation
"""
from sklearn.linear_model import LogisticRegression

def build_model():
    return LogisticRegression(max_iter=3000)
