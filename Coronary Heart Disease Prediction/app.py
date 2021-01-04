import utils
import model
# Data Loading and Numerical Operations
import pandas as pd
import numpy as np
# Metrics
from sklearn.metrics import precision_score, recall_score
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
                " on residents of the town of Framingham, Massachusetts. The classification goal is "
                "to predict whether the patient has 10-year risk of future coronary heart disease (CHD). The dataset "
                "provides the patients’ information. It includes over 4,240 records and 15 attributes."
                "")
    st.sidebar.markdown("Manually select the model you want to view and use the interactive text boxes, sliding bars "
                        "and buttons to tune the respective models. More than one options are provided for each model"
                        " and you can view and gain insight on how hyper-parameter tuning works. Enjoy exploring!")

    data = pd.read_csv("Dataset/framingham.csv")
    data = utils.preprocess(data)

    st.sidebar.markdown("\n#### Exploratory Data Analysis:")
    viz_list = st.sidebar.multiselect("",
                                      ('Categorical Visualisation',
                                       'Numerical Visualisation',
                                       'sysBP and diaBP Visualisation'))
    utils.visualize(viz_list, data)

    st.sidebar.markdown("\n#### Feature Selection:")
    feature = st.sidebar.radio("Feature selection using chi-squared test", ("Only Select Features",
                                                                            "Select Features and Plot"), key="feature")
    if feature == "Only Select Features":
        st.markdown("Best 10 features along with their plots")
        score, data = utils.feature_selection(data)
        st.write(score)
    else:
        score, data = utils.feature_selection(data)
        st.markdown("Best 10 features along with their plots")
        st.write(score)
        utils.plot_feature_selection(score)

    train_x, test_x, train_y, test_y = utils.split_and_scale(data)

    class_names = ["Has Heart Disease", "Doesn't have Heart Disease"]
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression",
                                                     "k-Nears Neighbour Classifier",
                                                     "Decision Tree Classifier",
                                                     "Random Forest Classifier",
                                                     "Gradient Boosting Classifier",
                                                     "XGBoost Classifier",
                                                     "Gaussian Naive Bayes Classifier"))

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='Lr')
        max_iter = st.sidebar.slider("Maximum no. of Iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regression Results")
            y_pred, accuracy, models = model.LR(train_x, test_x, train_y, test_y, C=C, max_iter=max_iter)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(test_y, y_pred, labels=class_names).round(3))
            st.write("Recall: ", recall_score(test_y, y_pred, labels=class_names).round(3))
            utils.plot_metrics(metrics, models, test_x, test_y, class_names)


    if classifier == "k-Nears Neighbour Classifier":
        st.sidebar.subheader("Model Hyperparameters")
        n = st.sidebar.number_input("n_neighbors (Number of nearest neighbors)", 1, 20, step=1, key='n')
        leaf_size = st.sidebar.slider("Leaf Size", 10, 200, key='leaf_size')
        algorithm = st.sidebar.radio("Algorithm to use", ("Ball Tree", "KD Tree", "Auto"), key='algorithm')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("kNN Classification Results")
            y_pred, accuracy, models = model.KNN(train_x, test_x, train_y, test_y, n=n, leaf_size=leaf_size,
                                                 algorithm=algorithm)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(test_y, y_pred, labels=class_names).round(3))
            st.write("Recall: ", recall_score(test_y, y_pred, labels=class_names).round(3))
            utils.plot_metrics(metrics, models, test_x, test_y, class_names)


    if classifier == "Decision Tree Classifier":
        criterion = st.sidebar.radio("Criterion of splitting trees", ("gini", "entropy"), key='criterion')
        max_depth = st.sidebar.slider("Max depth of the tree", 1, 50, key='amx_depth')
        min_samples_leaf = st.sidebar




if __name__ == '__main__':
    main()