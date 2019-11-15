from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn import svm
import pandas as pd
import sys, traceback
from sklearn import metrics
import numpy as np

def run_models(models, X, Y):
    scores = dict()
    for model in models:
        model.fit(X, Y)
        print(cross_val_score(model, X, Y, cv=5))
        score = np.mean(cross_val_score(model, X, Y, cv=5))*100
        scores[str(model.__class__)] = score
    return scores


def run():
    try:
        # Read features csv
        features = pd.read_csv('data/features.csv')
        # features = features.iloc[:60000,:] # TODO remove
        
        # Split dataset to train and test (80% & 20%)
        train_len = int(len(features)*0.8)
        train = features.iloc[:train_len, :]
        test = features.iloc[train_len:, :]

        x_train = train.iloc[:, 2:] # Distance Features
        y_train = train.iloc[:, 2:3] # Labels

        x_test = test.iloc[:, 2:]
        y_test = test.iloc[:, 2:3]

        # Handle Nan or inf values
        x_train = np.nan_to_num(x_train)
        y_train = np.nan_to_num(y_train)
        x_test = np.nan_to_num(x_test)
        y_test = np.nan_to_num(y_test)

        # log_reg(x_train, y_train, x_test, y_test)
        # support_vector_machine(x_train, y_train, x_test, y_test)
        # knn(x_train, y_train, x_test, y_test)
        # xgboost(x_train, y_train, x_test, y_test)
        kwargs = {
            "models": [
                svm.SVC(),
                XGBClassifier(),
                LogisticRegression(),
                KNeighborsClassifier(),
            ],
            "X": x_train,
            "Y": y_train,
        }
        scores = run_models(**kwargs)

        for model, score in scores.items():
            print("---------------------------")
            print("**** {} ****".format(model))
            print("Accuracy: {} %".format(score))
        print("---------------------------")


    except:
        print("!!! Exception caught")
        traceback.print_exc(file=sys.stdout)

run()
