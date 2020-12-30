# Data Loading and Numerical Operations
import pandas as pd
import numpy as np


def preprocess(data):
	# Filling the missing spaces of glucose column with the mode of the data (Mode = 75)
	data["glucose"].fillna((data["glucose"].mode())[0], inplace=True)

	# Drop all other null values
	data.dropna(inplace=True)

	# Remove outliers
	data = data[data['totChol']<600.0]
	data = data[data['sysBP']<295.0]

	return data


