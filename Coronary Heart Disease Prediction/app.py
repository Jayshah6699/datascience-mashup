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
        algorithm = st.sidebar.radio("Algorithm to use", ("ball_tree", "kd_tree", "auto"), key='algorithm')

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
        min_samples_leaf = st.sidebar.number_input("Minimum Leaf Samples", 1, 10, step=1, key='min_samples_leaf')
        max_features = st.sidebar.radio("No. of features to consider during best split", ("auto", "sqrt", "log2"),
                                        key='max_features')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Decision Tree Classification Results")
            y_pred, accuracy, models = model.DT(train_x, test_x, train_y, test_y, criterion=criterion, max_depth=max_depth,
                                leaf=min_samples_leaf, max_features=max_features)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(test_y, y_pred, labels=class_names).round(3))
            st.write("Recall: ", recall_score(test_y, y_pred, labels=class_names).round(3))
            utils.plot_metrics(metrics, models, test_x, test_y, class_names)



    if classifier == "Random Forest Classifier":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of Trees in the Random Forest", 100, 4000, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=5, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key='bootstrap')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest Classification Results")
            y_pred, accuracy, models = model.RF(train_x, test_x, train_y, test_y, n_estimators=n_estimators,
                           max_depth=max_depth, bootstrap=bootstrap)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(test_y, y_pred, labels=class_names).round(3))
            st.write("Recall: ", recall_score(test_y, y_pred, labels=class_names).round(3))
            utils.plot_metrics(metrics, models, test_x, test_y, class_names)


    if classifier == "Gradient Boosting Classifier":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of Trees in the Random Forest", 100, 4000, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=5, key='max_depth')
        learning_rate = st.sidebar.number_input("Learning Rate", 0.01, 10.0, step=0.01, key='learning_rate')
        warm_start = st.sidebar.radio("Reuse previous solution for more ensemble", ("True", "False"), key='warm_start')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Gradient Boosting Classification Results")
            y_pred, accuracy, models = model.GBC(train_x, test_x, train_y, test_y, n_estimators=n_estimators,
                           max_depth=max_depth, learning_rate=learning_rate, warm_start=warm_start)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(test_y, y_pred, labels=class_names).round(3))
            st.write("Recall: ", recall_score(test_y, y_pred, labels=class_names).round(3))
            utils.plot_metrics(metrics, models, test_x, test_y, class_names)


    if classifier == "XGBoost Classifier":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of Trees in the Random Forest", 100, 4000, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=5, key='max_depth')
        eta = st.sidebar.number_input("Learning Rate", 0.01, 10.0, step=0.01, key='eta')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))


if __name__ == '__main__':
    main()