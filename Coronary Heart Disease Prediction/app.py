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
st.set_page_config(page_title='Manual Parameter Tuner', layout='wide', initial_sidebar_state='auto')


def main():
    utils.local_css("css/styles.css")
    st.title("Heart Disease Prediction - Manual Parameter Tuner")
    st.sidebar.title("Manual Parameter Tuning")
    st.markdown("### Machine Learning is not only about the algorithms you use but also about the different Parameters "
                "assigned to each of them. The final model is heavily affected by the parameters used for a specific "
                "algorithm. "
                "\nThis interactive web app will help you to explore the various parameters of different ML algorithms."
                "\nThe different ML models presented here are:"
                "\n* Logistic Regression"
                "\n* Support Vector Classifier"
                "\n* k-Nears Neighbour Classifier"
                "\n* Decision Tree Classifier"
                "\n* Random Forest Classifier"
                "\n* Gradient Boosting Classifier"
                "\n* XGBoost Classifier"
                "\n### The dataset used here is the **Framingham** Coronary Heart Disease dataset publicly available "
                "at [Kaggle](https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset)."
                "\n## About the Dataset:"
                "\nThe **Framingham** dataset is from an ongoing cardiovascular study"
                " on residents of the town of Framingham, Massachusetts. The classification goal is "
                "to predict whether the patient has 10-year risk of future coronary heart disease (CHD). The dataset "
                "provides the patients’ information. It includes over 4,240 records and 15 attributes."
                "\n ### Even after optimizing parameters, the model would only work properly if accurate data is provided to it."
                " So, through this web app, the users will be able to get a feel of hyperparameter tuning but only on this specific dataset."
                "\n ## Head to the *Manual Parameter Tuning* section to get started!")

    st.sidebar.markdown("Manually select the model you want to view and use the interactive text boxes, sliding bars "
                        "and buttons to tune the respective models. More than one options are provided for each model"
                        " and you can view and gain insight on how hyper-parameter tuning works. Enjoy exploring!")

    data = pd.read_csv("Dataset/framingham.csv")
    data = utils.preprocess(data)

    st.sidebar.markdown("\n#### Exploratory Data Analysis:")
    viz_list = st.sidebar.multiselect("(Be sure to Clear off all the selected options here before moving on for faster response)",
                                      ('Categorical Visualisation',
                                       'Numerical Visualisation',
                                       'sysBP and diaBP Visualisation'))
    utils.visualize(viz_list, data)
    if st.sidebar.checkbox("View raw and preprocessed data", False):
        st.subheader("Raw-preprocessed Data")
        st.write(data)

    st.sidebar.markdown("\n#### Feature Selection:")
    feature = st.sidebar.radio("Feature selection using chi-squared test", ("Don't select features",
                                                                            "Select Features"), key='feature')
    if feature == "Don't select features":
        st.markdown("### Feature Selection is not done!")
    else:
        st.markdown("Best 10 features along with their chi-squared score")
        score, data = utils.feature_selection(data)
        st.write(score)
        if st.sidebar.checkbox("Plot Feature Selection", False):
            utils.plot_feature_selection(score)


    train_x, test_x, train_y, test_y = utils.split_and_scale(data)

    class_names = ["Has Heart Disease", "Doesn't have Heart Disease"]
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression",
                                                     "Support Vector Classifier",
                                                     "k-Nears Neighbour Classifier",
                                                     "Decision Tree Classifier",
                                                     "Random Forest Classifier",
                                                     "Gradient Boosting Classifier",
                                                     "XGBoost Classifier"))

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

    if classifier == "Support Vector Classifier":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        gamma = st.sidebar.radio("Gamma (for non linear hyperplanes)", ("auto", "scale"), key='gamma')
        kernel = st.sidebar.radio("Kernel (type of hyperplane)", ("linear", "rbf", "poly"), key='kernel')
        degree = 3
        if kernel == 'poly':
            degree = st.sidebar.number_input("Degree of the polynomial used to find the hyperplane", 1, 10, step=1,
                                             key='degree')
        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Classification Results")
            y_pred, accuracy, models = model.SVM(train_x, test_x, train_y, test_y, C=C, gamma=gamma, kernel=kernel,
                                                 degree=degree)
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
            y_pred, accuracy, models = model.DT(train_x, test_x, train_y, test_y, criterion=criterion,
                                                max_depth=max_depth,
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
        n_estimators = st.sidebar.slider("Number of Trees in the Gradient Boost ensemble", 100, 4000,
                                         key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=5, key='max_depth')
        learning_rate = st.sidebar.number_input("Learning Rate", 0.01, 10.0, step=0.01, key='learning_rate')
        warm_start = st.sidebar.radio("Reuse previous solution for more ensemble", ("True", "False"), key='warm_start')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Gradient Boosting Classification Results")
            y_pred, accuracy, models = model.GBC(train_x, test_x, train_y, test_y, n_estimators=n_estimators,
                                                 max_depth=max_depth, learning_rate=learning_rate,
                                                 warm_start=warm_start)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(test_y, y_pred, labels=class_names).round(3))
            st.write("Recall: ", recall_score(test_y, y_pred, labels=class_names).round(3))
            utils.plot_metrics(metrics, models, test_x, test_y, class_names)

    if classifier == "XGBoost Classifier":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of Trees in the XGBoost ensemble", 100, 4000, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=5, key='max_depth')
        eta = st.sidebar.number_input("Learning Rate", 0.01, 10.0, step=0.01, key='eta')
        colsample_bytree = st.sidebar.number_input("Percentage of features used per tree", 0.01, 1.0, step=0.01,
                                                   key='colsample_bytree')
        reg_alpha = st.sidebar.number_input("L1 regularization on leaf weights", 1, 10, step=1, key='reg_alpha')
        reg_lambda = st.sidebar.number_input("L2 regularization on leaf weights", 1, 10, step=1, key='reg_lambda')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Extreme Gradient Boosting(XGBoost) Classification Results")
            y_pred, accuracy, models = model.XGB(train_x, test_x, train_y, test_y, n_estimators=n_estimators,
                                                 max_depth=max_depth, eta=eta, colsample_bytree=colsample_bytree,
                                                 reg_alpha=reg_alpha, reg_lambda=reg_lambda)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(test_y, y_pred, labels=class_names).round(3))
            st.write("Recall: ", recall_score(test_y, y_pred, labels=class_names).round(3))
            utils.plot_metrics(metrics, models, test_x, test_y, class_names)


if __name__ == '__main__':
    main()
