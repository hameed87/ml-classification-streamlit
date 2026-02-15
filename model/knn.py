"""
KNN model code

Author: Abdul
Date: February 10, 2026
Description: K-Nearest Neighbors classification model implementation
"""
from sklearn.neighbors import KNeighborsClassifier

def build_model():
    return KNeighborsClassifier(n_neighbors=15)
