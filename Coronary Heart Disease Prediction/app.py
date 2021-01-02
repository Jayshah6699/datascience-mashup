import utils
# Data Loading and Numerical Operations
import pandas as pd
import numpy as np
# Metrics
from sklearn.metrics import precision_score, recall_score
# Data Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
# Web App
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
    utils.local_css("css/styles.css")
    st.title("Heart Disease Prediction - Manual Parameter Tuning Visualizer")
    st.sidebar.title("Manual Parameter Tuning")
    st.markdown("### Machine Learning is not only about the algorithms you use but also about the Parameters you assign"
                "to each of them. Your final model is heavily affected by the parameters used in a specific algorithm. "
                "\nThis interactive web app will help you explore various parameters of different ML algorithms."
                "\nThe different ML models presented here are:"
                "\n* Logistic Regression"
                "\n* k-Nears Neighbour Classifier"
                "\n* Decision Tree Classifier"
                "\n* Random Forest Classifier"
                "\n* Gradient Boosting Classifier"
                "\n* XGBoost Classifier"
                "\n* Gaussian Naive Bayes Classifier"
                "\n### The dataset used here is the **Framingham** dataset publicly available "
                "at [Kaggle](https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset)."
                "\n## About the Dataset:"
                "\nThe **Framingham** dataset is from an ongoing cardiovascular study"
                "on residents of the town of Framingham, Massachusetts. The classification goal is "
                "to predict whether the patient has 10-year risk of future coronary heart disease (CHD).The dataset "
                "provides the patients’ information. It includes over 4,240 records and 15 attributes."
                "")
    st.sidebar.markdown("Manually select the model you want to view and use the interactive text boxes, sliding bars "
                        "and buttons to tune the respective models. More than one options are provided for each model"
                        " and you can view and gain insight on how hyper-parameter tuning works. Enjoy exploring!")

    data = pd.read_csv("Dataset/framingham.csv")
    data = utils.preprocess(data)

    viz_list = st.sidebar.multiselect("Exploratory Data Analysis:",
                                      ('Categorical Visualisation',
                                       'Numerical Visualisation',
                                       'sysBP and diaBP Visualisation'))
    utils.visualize(viz_list, data)


if __name__ == '__main__':
    main()