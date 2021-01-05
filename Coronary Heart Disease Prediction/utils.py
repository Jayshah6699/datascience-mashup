# Dataset and Matrix Manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

sns.set_context("paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20})


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache(persist=True)
def preprocess(data):
    # Fill the missing spaces of glucose column with the mode of the data (Mode = 75)
    data["glucose"].fillna((data["glucose"].mode())[0], inplace=True)

    # Drop all other null values
    data.dropna(inplace=True)

    # Remove outliers
    data = data[data['totChol'] < 600.0]
    data = data[data['sysBP'] < 295.0]

    # Resample data to reduce imbalance
    target1 = data[data['TenYearCHD'] == 1]
    target0 = data[data['TenYearCHD'] == 0]
    target1 = resample(target1, replace=True, n_samples=len(target0), random_state=40)
    data = pd.concat([target0, target1])

    return data


def visualize(viz_list, data):
    if "Categorical Visualisation" in viz_list:
        st.subheader("Distribution of all Categorical features")
        categorical_features = ['male', 'education', 'currentSmoker', 'BPMeds',
                            'prevalentStroke', 'prevalentHyp', 'diabetes']
        num_plots = len(categorical_features)
        tot_cols = 2
        tot_rows = num_plots//tot_cols + 1
        fig, axs = plt.subplots(nrows=tot_rows, ncols=tot_cols, figsize=(7*tot_cols, 7*tot_rows),
                            facecolor='w', constrained_layout=True)
        for i, var in enumerate(categorical_features):
            row = i // tot_cols
            pos = i % tot_cols
            sns.countplot(x=var, data=data, ax=axs[row][pos])
        st.pyplot()

    if "Numerical Visualisation" in viz_list:
        st.subheader("Distribution of all numerical features")
        numeric_features = ['cigsPerDay', 'totChol', 'sysBP', 'diaBP',
                        'BMI', 'heartRate', 'glucose']
        num_plots = len(numeric_features)
        tot_cols = 2
        tot_rows = num_plots // tot_cols + 1
        fig, axs = plt.subplots(nrows=tot_rows, ncols=tot_cols, figsize=(7 * tot_cols, 7 * tot_rows),
                            facecolor='w', constrained_layout=True)
        for i, var in enumerate(numeric_features):
            row = i // tot_cols
            pos = i % tot_cols
            sns.kdeplot(x=var, data=data, ax=axs[row][pos])
        st.pyplot()


    if 'sysBP and diaBP Visualisation' in viz_list:
        st.subheader("TenYearCHD Distribution of sysBP and diaBP with respect to currentSmoker and gender")
        sns.lmplot(x='sysBP', y='diaBP', data=data, hue='TenYearCHD',
                   col='male', row='currentSmoker')
        st.pyplot()


@st.cache(persist=True)
def feature_selection(data):
    X = data.iloc[:, 0:15]
    y = data.iloc[:, -1]

    best_features = SelectKBest(score_func=chi2, k=10).fit(X, y)

    data_scores = pd.DataFrame(best_features.scores_)
    data_columns = pd.DataFrame(X.columns)

    scores = pd.concat([data_columns, data_scores], axis=1)
    scores.columns = ['Feature', 'Score']
    scores = scores.sort_values(by="Score", ascending=False)

    return scores, data[['sysBP', 'glucose', 'age', 'cigsPerDay', 'totChol', 'diaBP',
                 'prevalentHyp', 'male', 'BPMeds', 'diabetes', 'TenYearCHD']]


def plot_feature_selection(scores):
    st.subheader("Chi-squared Score distribution of the best 10 features")
    plt.figure(figsize=(12, 7), facecolor='w')
    sns.barplot(x='Feature', y='Score', data=scores, palette='BuGn_r')
    plt.xticks(rotation=45)
    st.pyplot()


@st.cache(persist=True)
def split_and_scale(data):
    y = data['TenYearCHD'].values
    X = data.drop(['TenYearCHD'], axis=1).values
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