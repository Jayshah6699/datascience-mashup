import utils
# Data Loading and Numerical Operations
import pandas as pd
import numpy as np
# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Hyper-parameter Tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# Ensemble
from mlxtend.classifier import StackingCVClassifier


def LR(X_train, X_test, y_train, y_test, C, max_iter):
    lr = LogisticRegression(C=C, max_iter=max_iter)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = lr.score(X_test, y_test)
    return y_pred, accuracy


def KNN(X_train, X_test, y_train, y_test, n):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)
    return y_pred, accuracy


def DT(X_train, X_test, y_train, y_test, criterion, max_depth, leaf, split):
    dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                min_samples_leaf=leaf, min_samples_split=split)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)
    return y_pred, accuracy


def RF(X_train, X_test, y_train, y_test, n_estimators, max_depth, bootstrap):
    rf = RandomForestClassifier(n_estimators=n_estimators,
                           max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)
    return y_pred, accuracy


def GBC(X_train, X_test, y_train, y_test, n_estimators, max_depth, learning_rate, warm_start):
    gbc = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     learning_rate=learning_rate, warm_start=warm_start)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    accuracy = gbc.score(X_test, y_test)
    return y_pred, accuracy