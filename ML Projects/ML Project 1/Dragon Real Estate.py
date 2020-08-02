#!/usr/bin/env python
# coding: utf-8

# # Dragon Real Estate Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv('data1.csv')


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS']


# In[6]:


housing['CHAS'].value_counts()


# In[7]:


housing=housing.dropna()


# In[8]:


housing.info()


# In[9]:


housing.describe()


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


housing.hist(bins=50,figsize=(20,15))


# ## Train-Test Splitting

# In[12]:


#for learning pupose
import numpy as np
def split_test_train(data,test_ratio):
    np.random.seed(42)
    shuffle=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffle[:test_set_size]
    train_indices=shuffle[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[13]:


#train_set,test_set=split_test_train(housing,0.2)


# In[14]:


#print("Rows in train set:",len(train_set),"\nRows in test set:",len(test_set))


# In[15]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)


# In[16]:


print("Rows in train set:",len(train_set),"\nRows in test set:",len(test_set))


# In[17]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.iloc[train_index]
    strat_test_set=housing.iloc[test_index]


# In[18]:


strat_train_set


# In[19]:


strat_test_set


# In[20]:


strat_train_set['CHAS'].value_counts()


# In[21]:


strat_test_set['CHAS'].value_counts()


# In[22]:


331/28


# In[23]:


83/7


# ## Looking for Corelations

# In[24]:


corr_matrix=housing.corr()


# In[25]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[26]:


from pandas.plotting import scatter_matrix
attributes=['MEDV','RM','ZN','LSTAT']
scatter_matrix(housing[attributes],figsize=(12,8))


# In[27]:


housing.plot(kind='scatter',x='RM',y='MEDV',alpha=0.8)


# # Trying out Attributes Combination

# In[28]:


housing['TAXRM']=housing['TAX']/housing['RM']


# In[29]:


housing['TAXRM']


# In[30]:


housing.head()


# In[31]:


corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[32]:


housing.plot(kind='scatter',x='TAXRM',y='MEDV',alpha=0.8)


# In[33]:


housing=strat_train_set.drop('MEDV',axis=1)
housing_labels=strat_train_set['MEDV'].copy()


# # Missing Attributes

# In[34]:


# To take care of missing value in data you have 3 choices-
#     1. Get rid of missing data points.
#     2. Get rid of whole attribute.
#     3. Set missing value to some value(0,mean,median)


# In[35]:


a=housing.dropna(subset=['RM'])   # Option 1
a.shape
# Note that original housing data will remain unchanged


# In[36]:


housing.drop('RM',axis=1)        # Option 2
#note RM is removed but original housing data is unchanged


# In[37]:


housing.head()


# In[38]:


median=housing['RM'].median
housing['RM'].fillna(median)        #     Option 3
# Note that original housing dataframe will remain unchanged


# In[39]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
imputer.fit(housing)


# In[40]:


imputer.statistics_


# In[41]:


X=imputer.transform(housing)


# In[42]:


housing_tr=pd.DataFrame(X,columns=housing.columns)


# In[43]:


housing.describe()


# ## Scikit-Learn Design

# Primarily, three types of objects,
#     1. Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters.
#     2. Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.
#     3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.

# ## Feature Scaling

# Primarily, two types of feature scaling methods:
#     1. Min-max scaling (Normalization)
#         (value - min)/(max - min)
#         Sklearn provides a class called MinMaxScaler for this
#         
#     2. Standardization
#         (value - mean)/std
#         Sklearn provides a class called StandardScaler for this

# In[44]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    #.... add as many as you want in your pipeline
    ('std_scaler',StandardScaler()),
])


# In[45]:


housing_num_tr=my_pipeline.fit_transform(housing)


# In[46]:


housing_num_tr   #it is a numpy array


# # Selecting a desired model for Dragon Real Estates

# In[47]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=DecisionTreeRegressor()
# model=LinearRegression()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[48]:


some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[49]:


list(some_labels)


# # Evaluating the model

# In[50]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)
rmse


# ## Using better evaluation technique- Cross Validation

# In[51]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring='neg_mean_squared_error',cv=10)
rmse_scores=np.sqrt(-scores)
rmse_scores


# In[52]:


def print_scores(scores):
    print("Scores - ",scores)
    print('Mean Scores - ',scores.mean())
    print('Standard deviation scores - ',scores.std())


# In[53]:


print_scores(rmse_scores)


# ## Saving the Model

# In[54]:


from joblib import dump,load
dump(model,'Dragon.joblib')


# ## Testing the Model on test data

# In[55]:


X_test=strat_test_set.drop('MEDV',axis=1)
Y_test=strat_test_set['MEDV'].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_prediction=model.predict(X_test_prepared)
final_mse=mean_squared_error(final_prediction,Y_test)
final_rmse=np.sqrt(final_mse)
final_rmse


# In[56]:


prepared_data


# ## Using the model

# In[57]:


import numpy as np
features=np.array([[-0.30380946, -0.51697019, -0.35100881, -0.29084729, -0.03681906,
        -0.78602927,  0.87313616,  0.32251744, -0.5019941 , -0.47809192,
         1.26243039,  0.10355559,  0.50455981]])
model.predict(features)


# In[ ]:




