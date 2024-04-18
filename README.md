# **License Status Prediction: Multi-label Classification Approach with KNN, Naive Bayes, Logistic Regression, Decision Trees, Random Forest, AdaBoost, and XGBoost**

### Objective
To predict license status for the given business with multi label classification.

## Data Description
The dataset used is a licensed dataset. It contains information about 86K different businesses over various features. The target variable is the status of license which has five different categories.
- **Target Variable Classifications:**
1. AAI - License status is issued
2. AAC - License status is cancelled
3. REV - License status is revoked
4. REA - License status is revoked and appealed
5. INQ - License status is in enquiry

## Tech Stack
- Language: Python
- Libraries: pandas, scikit_learn, category_encoders, numpy, os, seaborn,
matplotlib, hyperopt, xgboost

## Approach
- Data Description
- Exploratory Data Analysis
- Data Cleaning
  - Missing Value imputation
  - Outlier Detection
- Data Imbalance
- Data Encoding
- Model Building
  - KNN classifier
  - Naive Bayes algorithm
  - Logistic Regression
  - Decision Tree classifier
  - Random Forest
  - AdaBoost
  - XGBoost
- Classification Metrics
  - Precision
  - Recall
  - F1 score
  - Accuracy
  - Macro average
  - Weighted average
- Feature importance
- Hyperparameter Tuning
  - Random Search Optimisation
  - Grid Search Optimisation
  - Bayesian Optimisation

## Project Structure
```
|-- InputFiles
    -- License_data.csv
|-- SourceFolder
    |-- ML_Pipeline
        -- model_selection.py
        -- preprocessing.py
    |-- Engine.py

