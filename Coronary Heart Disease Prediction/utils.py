# Dataset and Matrix Manipulation
import pandas as pd
import numpy as np
# Data Resampling
from sklearn.utils import resample
# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Data Splitting
from sklearn.model_selection import train_test_split
# Data Scaling
from sklearn.preprocessing import MinMaxScaler
# Metrics
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
# Web App
import streamlit as st



@st.cache(persist=True)
def preprocess(data):
    # Fill the missing spaces of glucose column with the mode of the data (Mode = 75)
    data["glucose"].fillna((data["glucose"].mode())[0], inplace=True)

    # Drop all other null values
    data.dropna(inplace=True)

    # Remove outliers
    data = data[data['totChol']<600.0]
    data = data[data['sysBP']<295.0]

    # Resample data to reduce imbalance
    target1 = data[data['TenYearCHD'] == 1]
    target0 = data[data['TenYearCHD'] == 0]
    target1 = resample(target1, replace=True, n_samples=len(target0), random_state=40)
    data = pd.concat([target0, target1])

    return data


def feature_selection(data):
    X = data.iloc[:, 0:15]
    y = data.iloc[:, -1]

    best_features = SelectKBest(score_func=chi2, k=10).fit(X, y)

    data_scores = pd.DataFrame(best_features.scores_)
    data_columns = pd.DataFrame(X.columns)

    scores = pd.concat([data_columns, data_scores], axis=1)
    scores.columns = ['Feature', 'Score']

    # Select 10 features
    features = scores["Feature"].tolist()[:10]

    return data[features]


@st.cache(persist=True)
def split_and_scale(data):
    y = data['TenYearCHD']
    X = data.drop(['TenYearCHD'], axis=1)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=1)

    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    return train_x, test_x, train_y, test_y


def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()