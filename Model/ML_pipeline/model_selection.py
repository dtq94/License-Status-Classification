import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import category_encoders as ce
from sklearn.neighbors import KNeighborsClassifier
# Metrics Performance
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


#Function to  compare various models and finding the best one
def run_models(X_train, y_train, X_test, y_test, model_type = 'Imbalanced'):
    
    clfs = {'KNNClassifier': KNeighborsClassifier(n_neighbors=3),
            'LogisticRegression' : LogisticRegression(),
            'GaussianNB': GaussianNB(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'XGBoostClassifier': XGBClassifier()           
            }
    cols = ['model','precision_score', 'recall_score','f1_score']

    models_report = pd.DataFrame(columns = cols)
    conf_matrix = dict()

    for clf, clf_name in zip(clfs.values(), clfs.keys()):

        if clf == "KNNClassifier":
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train  = scaler.transform(X_train)
            scaler.fit(X_test)
            X_test = scaler.transform(X_test)


        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:,1]

        print('computing {} - {} '.format(clf_name, model_type))

        tmp = pd.Series({'model_type': model_type,
                         'model': clf_name,                       
                         'precision_score': metrics.precision_score(y_test, y_pred,average='macro'),
                         'recall_score': metrics.recall_score(y_test, y_pred,average='macro'),
                         'f1_score': metrics.f1_score(y_test, y_pred,average='macro')})

        models_report = models_report.append(tmp, ignore_index = True)    
    print(models_report.sort_values(by=["f1_score"]))
    parent_dir = os.path.dirname(os.getcwd())
    data_path = "\\output\\"
    models_report.to_excel(parent_dir+data_path+"model_report.xlsx")
    return models_report

