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


def logistic(X_train, X_test, y_train, y_test, C, max_iter):
    lr = LogisticRegression(C=C, max_iter=max_iter)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = lr.score(X_test, y_test)
    return y_pred, accuracy


def KNN(X_train, X_test, y_train, y_test, n):
    knn = KNeighborsClassifier()