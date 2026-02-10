# Machine Learning Classification Models with Streamlit Deployment

---

## a. Problem Statement

The objective of this project is to design, implement, and evaluate multiple machine learning classification models on a real-world dataset. The project demonstrates an end-to-end machine learning workflow that includes data preprocessing, model training, evaluation using multiple performance metrics, and deployment of an interactive web application using Streamlit.

---

## b. Dataset Description  **[1 Mark]**

<!-- TODO: Fill in dataset information -->
- **Dataset Name**:   
- **Dataset Source**:   
- **Problem Type**:   
- **Number of Instances**:  
- **Number of Features**:   

### Dataset Details

The dataset consists of multiple numerical and categorical features related to patient health indicators. These features are used to predict the target variable, which represents the presence or absence of a specific condition. Prior to model training, the dataset was cleaned and preprocessed to ensure consistency and suitability for machine learning algorithms.

---

## c. Models Used and Evaluation Metrics  **[6 Marks]**

The following six machine learning classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model was evaluated using the following performance metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

### Comparison Table of Model Performance
<!-- TODO: Fill in Model Performance -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| K-Nearest Neighbors |  |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest (Ensemble) |  |  |  |  |  |  |
| XGBoost (Ensemble) |  |  |  |  |  |  |

---

## d. Observations on Model Performance  **[3 Marks]**

<!-- TODO: Fill in observations for all models -->
| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression |  |
| Decision Tree |  |
| K-Nearest Neighbors |  |
| Naive Bayes |  |
| Random Forest (Ensemble) |  |
| XGBoost (Ensemble) |  |

---

## e. Streamlit Web Application

An interactive web application was developed using Streamlit and deployed on Streamlit Community Cloud.

### Application Features

- CSV dataset upload option (test data only)  
- Model selection using a dropdown menu  
- Display of evaluation metrics  
- Confusion matrix and classification report visualization  

<!-- TODO: Add Streamlit app deployment link here -->
**Live Streamlit App Link**:  


---

## f. GitHub Repository Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   │-- logistic_regression.py
│   │-- decision_tree.py
│   │-- knn.py
│   │-- naive_bayes.py
│   │-- random_forest.py
│   │-- xgboost.py
```

---

## g. Deployment Details

The application was deployed using Streamlit Community Cloud by connecting the GitHub repository and selecting app.py as the main application file. All required dependencies for the project are listed in the requirements.txt file to ensure successful deployment.

---

## h. BITS Virtual Lab Execution Proof

The complete implementation and execution of this assignment were performed on the BITS Virtual Lab environment. A screenshot showing successful execution has been included in the final PDF submission as proof of compliance.

---

### End of README
