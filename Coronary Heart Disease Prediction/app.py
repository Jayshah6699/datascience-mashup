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


data = pd.read_csv("Dataset/framingham.csv")

data = utils.preprocess(data)

print(data.shape, data.head())

print(utils.feature_selection(data))