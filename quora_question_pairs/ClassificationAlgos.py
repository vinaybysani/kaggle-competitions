from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pandas as pd
import sys, traceback
from sklearn import metrics
import numpy as np


def support_vector_machine(x_train, y_train, x_test, y_test):
    '''
    SVM
    '''
    # Low accuracy on low dataset, time taking for entire dataset, need to try
    # Also, need to see how to vary params

    svm_model = svm.SVC()
    svm_model.fit(x_train, y_train)
    predicted = svm_model.predict(x_test)
    print("**** Support Vector Machine ****")
    print("Training size: ",len(x_train))
    print("Test size: ",len(x_test))
    print("Accuracy:",metrics.accuracy_score(y_test, predicted)*100,"%")


def log_reg(x_train, y_train, x_test, y_test):
    '''
    Logistic Regression 
    '''
    # Logistic regression model (64% on 80-20 split)

    logmodel = LogisticRegression()
    logmodel.fit(x_train, y_train)
    predicted = logmodel.predict(x_test)
    print("**** Logistic Regression ****")
    print("Training size: ",len(x_train))
    print("Test size: ",len(x_test))
    print("Accuracy:",metrics.accuracy_score(y_test, predicted)*100,"%")


def knn(x_train, y_train, x_test, y_test):
    '''
    K Nearest Neighbours
    '''
    # 71% accuracy on k=5(default)
    # Need to try with multiple 'k' values

    knn_model = KNeighborsClassifier()
    knn_model.fit(x_train, y_train)
    predicted = knn_model.predict(x_test)
    print("**** K Nearest Neighbours ****")
    print("Training size: ",len(x_train))
    print("Test size: ",len(x_test))
    print("Accuracy:",metrics.accuracy_score(y_test, predicted)*100,"%")


def run():
    try:
        # Read features csv
        features = pd.read_csv('data/features.csv')
        # features = features.iloc[:60000,:] # TODO remove
        
        # Split dataset to train and test (80% & 20%)
        train_len = int(len(features)*0.8)
        train = features.iloc[:train_len, :]
        test = features.iloc[train_len:, :]

        x_train = train.iloc[:, 3:] # Distance Features
        y_train = train.iloc[:, 2:3] # Labels

        x_test = test.iloc[:, 3:]
        y_test = test.iloc[:, 2:3]

        # Handle Nan or inf values
        x_train = np.nan_to_num(x_train)
        y_train = np.nan_to_num(y_train)
        x_test = np.nan_to_num(x_test)
        y_test = np.nan_to_num(y_test)

        # log_reg(x_train, y_train, x_test, y_test)
        # support_vector_machine(x_train, y_train, x_test, y_test)
        knn(x_train, y_train, x_test, y_test)

    except:
        print("!!! Exception caught")
        traceback.print_exc(file=sys.stdout)

run()
