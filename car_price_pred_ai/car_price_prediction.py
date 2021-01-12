#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('car_data.csv')


# In[3]:


df.head()


# In[5]:


df.shape


# In[6]:


print(df['Seller_Type'].unique())


# In[26]:


print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df['Fuel_Type'].unique())


# In[8]:


# check missing or null values
df.isnull().sum()


# In[9]:


df.describe()


# In[11]:


df.columns


# In[12]:


final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[13]:


final_dataset.head()


# In[14]:


final_dataset['Current_Year']=2020


# In[15]:


final_dataset.head()


# In[16]:


final_dataset['no_of_year']=final_dataset['Current_Year']-final_dataset['Year']


# In[17]:


final_dataset.head()


# In[19]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[20]:


final_dataset.head()


# In[21]:


final_dataset.drop(['Current_Year'],axis=1,inplace=True)


# In[22]:


final_dataset.head()


# In[30]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[31]:


final_dataset.head()


# In[32]:


final_dataset.corr()


# In[33]:


import seaborn as sns


# In[34]:


sns.pairplot(final_dataset)


# In[35]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))

#plot heat map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[59]:


final_dataset.head()


# In[60]:


# independent and dependent features
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[61]:


X.head()


# In[62]:


y.head()


# In[63]:


# ordering of features importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)


# In[64]:


print(model.feature_importances_)


# In[65]:


# plot graph of feature importance for visualzation
feat_importances=pd.Series(model.feature_importances_,index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[67]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[69]:


X_train.shape


# In[70]:


from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()


# In[78]:


# Hyperparameters
# Randomized Search CV
# Number Of trees in  random forest
import numpy as np

n_estimators=[int(x) for x in np.linspace(start = 100,stop = 1200,num = 12)]

#Number of features to consider at every split
max_features=['auto','sqrt']

# Maximum number of levels in a tree
max_depth =[int(x) for x in np.linspace(5, 30,num =6)]

# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split =[2,5,10,15,100]

# Minimum number of samples required to split each leaf node
min_samples_leaf = [1,2,5,10]


# In[79]:


from sklearn.model_selection import RandomizedSearchCV


# In[80]:


# create random grid
random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf}

print(random_grid)


# In[83]:


# use random grid  to search for best hyperparameters
# first create the base model to tune
rf=RandomForestRegressor()


# In[85]:


rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,scoring ='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs =1)


# In[86]:


rf_random.fit(X_train,y_train)


# In[87]:


predictions = rf_random.predict(X_test)


# In[88]:


predictions


# In[89]:


sns.distplot(y_test-predictions)


# In[90]:


plt.scatter(y_test,predictions)


# In[92]:


import pickle
# open a file where you want to store data
file=open('random_forest_regression_model.pkl' , 'wb')

#dump information to that file
pickle.dump(rf_random,file)


# In[ ]:





# In[ ]:





# In[ ]:




