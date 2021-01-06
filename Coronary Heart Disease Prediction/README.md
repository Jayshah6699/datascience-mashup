<h1 align='center'>
  Heart Disease Prediction - Manual Parameter Tuner
</h1>

<h3 align="center"> This interactive web app is designed to help beginners in Machine Learning and Data Science to explore the various parameters of different ML algorithms out there.</h3> 

[![UI ](https://img.shields.io/badge/Website-Link%20to%20Web%20App-orange?style=for-the-badge&logo=appveyor)](https://share.streamlit.io/indrap24/manual-parameter-tuner/main/app.py)

### Functionalities
- [x]  Manually tune parameters of different ML algorithms to get the best result on the **Framingham** Heart Disease dataset.

### The different ML models presented here are:

* Logistic Regression
* Support Vector Classifier
* k-Nears Neighbour Classifier
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier


### About the Dataset:
The dataset used here is the **Framingham** Coronary Heart Disease dataset publicly available at [Kaggle](https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset).

The Framingham dataset is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD). The dataset provides the patients’ information. It includes over 4,240 records and 15 attributes.

Even after optimizing parameters, the model would only work properly if accurate data is provided to it. So, through this web app, will help the users to be able to get a feel of hyperparameter tuning but only on *this specific dataset.*


### Check out the published Website here:

https://share.streamlit.io/indrap24/manual-parameter-tuner/main/app.py




<br>

### Instructions to run the web app locally
  
* Pre-requisites:
	-  Python 3.6 or 3.7 or 3.8
	-  Dependencies from requirements.txt
  
* Directions to Install

   - First clone this repository onto your system.<br>
   - Then, create a Virtual Environment and install the packages from requirements.txt: <br>
   - Navigate to this repository, create a Virtual Environment and activate it: <br>
   ```bash
  cd path/to/cloned/repo
  python3 -m venv env
  source env/bin/activate
  ```
  Install the python dependencies from requirements.txt:
    ```bash
    pip install requirements.txt
     ```
* Directions to Execute

    From anywhere in the project directory, run the following command in the terminal -
    ```bash
    streamlit run app.py
    ```
    
    This will prompt a localhost and you can view and make changes to the source file locally.

