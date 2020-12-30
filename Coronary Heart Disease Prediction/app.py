import utils
# Data Loading and Numerical Operations
import pandas as pd
import numpy as np
# Data Visualizations
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("Dataset/framingham.csv")

data = utils.preprocess(data)

print(data.shape, data.head())

print(utils.feature_selection(data))