<h1 align="center">TMDb Movie Analysis and Comparative Modeling</h1>


### Abstract:

In this machine learning project, I cleaned, analyzed, and predicted two target variables – both **revenue**
(numerical) and **profitability** (categorical), from the dataset of **THE MOVIE DATABASE (TMDb).**

### Aim: 

To explore the various Data Processing, Analysis and Regression and Classification Modeling
techniques required for the Dataset to provide better predictions of the Revenue or the Profitability of
a movie before its production.

### Dataset:

The [movie dataset on which this case study is based](https://www.kaggle.com/tmdb/tmdb-movie-metadata) is a database of 5000 movies catalogued by [The Movie Database (TMDb)](https://www.themoviedb.org/?language=en). The information available about each movie include its budget, revenue generated, genres, rating, vote
count, popularity, actors and actresses and any more. However, I used an [unclean version](https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@movie_data.csv) of the dataset for this project from a HarvardX Course asset.


### Implementation:

In this project, I will use this dataset to **clean, analyze and determine** whether any information about
a movie can predict the total revenue of a movie. I will then attempt to predict whether a movie's
revenue will exceed its budget (profitability). Also, I will test **two** different models for each
prediction to check which predicts our target variable better. To sum it up, 
The project is divided into 3 parts:
* Data Preprocessing:
  * Importing Libraires
  * Reading the Dataset
  * Defining feature and target variables
  * Removing null values
  * Feature Engineering - Feature Selection and Transformation
  * Storing the transformed Dataset
* Exploratory Data Analysis:
  * Descriptive Statistical Analysis
  * Data Visualizations
  * Data Trend Analysis
* Predictive Modelling & Evaluation (Using 10-fold Cross Validation): 
  * For Predicting Revenue (Regression):
    * Linear Regression
    * Random Forest Regression
  * For Predicting Profitability (Classification):
    * Logistic Regression
    * Random Forest Classification
  * Evaluation and Analysis


The results obtained from this project will be helpful for the *Movie Production Teams* to analyze the
rubrics of their Movie Idea before it moves on to the Production Phase.
