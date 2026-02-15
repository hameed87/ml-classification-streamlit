# ML Assignment 2 – Credit Card Default Prediction

## 🔹 Problem Statement

The objective of this project is to build multiple machine learning classification models to predict whether a credit card customer will default on payment next month using historical billing and payment data.
The project also includes deployment of an interactive Streamlit web application for model demonstration and evaluation.

---

## 🔹 Dataset Description

Dataset: Default of Credit Card Clients (UCI / Kaggle)
Instances: 30,000
Features: 23
Target variable: `default.payment.next.month`

The dataset contains demographic information, credit limit, past payment history, bill statements, and payment amounts for credit card customers.
The task is a binary classification problem to predict whether a customer will default on payment in the next month.

---

## 🔹 Models Implemented

The following classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

---

## 🔹 Evaluation Metrics

Each model was evaluated using:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

## 🔹 Model Comparison Table

| ML Model Name       | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| ------------------- | -------- | --------- | --------- | ------ | -------- | --------- |
| Logistic Regression | 0.8109   | 0.7347    | 0.7203    | 0.2375 | 0.3572   | 0.3373    |
| Decision Tree       | 0.7324   | 0.6200    | 0.3998    | 0.4183 | 0.4088   | 0.2361    |
| KNN                 | 0.8151   | 0.7500    | 0.6581    | 0.3412 | 0.4494   | 0.3789    |
| Naive Bayes         | 0.5971   | 0.7434    | 0.3251    | 0.7637 | 0.4561   | 0.2604    |
| Random Forest       | 0.8255   | 0.7843    | 0.6842    | 0.3918 | 0.4983   | 0.4248    |
| XGBoost             | 0.8277   | 0.7990    | 0.7041    | 0.3816 | 0.4949   | 0.4294    |

---

## 🔹 Observations on Model Performance

| ML Model            | Observation                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Provides a strong baseline with good accuracy but lower recall, indicating difficulty in identifying all defaulters. |
| Decision Tree       | Captures nonlinear patterns but shows lower overall performance and MCC due to overfitting.                          |
| KNN                 | Performs moderately well but sensitive to feature scaling and hyperparameters.                                       |
| Naive Bayes         | High recall but low accuracy due to independence assumptions among features.                                         |
| Random Forest       | Strong ensemble performance with improved MCC and better stability.                                                  |
| XGBoost             | Best performing model overall with highest AUC and MCC, capturing complex patterns effectively.                      |

---

## 🔹 Streamlit App Features

The deployed Streamlit application provides:

* CSV dataset upload option
* Model selection dropdown
* Display of evaluation metrics
* Confusion matrix
* Classification report

---

## 🔹 How to Run Locally

```bash
pip install -r requirements.txt
python -m model.train_and_save
streamlit run app.py
```

---

## 🔹 Live Streamlit App

```

```

---

## 🔹 GitHub Repository

```

```
