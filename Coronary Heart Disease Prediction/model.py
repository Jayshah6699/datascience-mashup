# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


def LR(X_train, X_test, y_train, y_test, C, max_iter):
    lr = LogisticRegression(C=C, max_iter=max_iter)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = lr.score(X_test, y_test)
    return y_pred, accuracy, lr


def SVM(X_train, X_test, y_train, y_test, C, gamma, kernel, degree):
    svc = SVC(C=C, gamma=gamma, kernel=kernel, degree=degree)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    accuracy = svc.score(X_test, y_test)
    return y_pred, accuracy, svc


def KNN(X_train, X_test, y_train, y_test, n, leaf_size, algorithm):
    knn = KNeighborsClassifier(n_neighbors=n, leaf_size=leaf_size, algorithm=algorithm)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)
    return y_pred, accuracy, knn


def DT(X_train, X_test, y_train, y_test, criterion, max_depth, leaf, max_features):
    dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                min_samples_leaf=leaf, max_features=max_features)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)
    return y_pred, accuracy, dt


def RF(X_train, X_test, y_train, y_test, n_estimators, max_depth, bootstrap):
    rf = RandomForestClassifier(n_estimators=n_estimators,
                           max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)
    return y_pred, accuracy, rf


def GBC(X_train, X_test, y_train, y_test, n_estimators, max_depth, learning_rate, warm_start):
    gbc = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     learning_rate=learning_rate, warm_start=warm_start)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    accuracy = gbc.score(X_test, y_test)
    return y_pred, accuracy, gbc


def XGB(X_train, X_test, y_train, y_test, eta, max_depth, n_estimators, colsample_bytree, reg_alpha, reg_lambda):
    xgb = XGBClassifier(eta=eta, max_depth=max_depth, n_estimators=n_estimators, colsample_bytree=colsample_bytree,
                        reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    accuracy = xgb.score(X_test, y_test)
    return y_pred, accuracy, xgb
