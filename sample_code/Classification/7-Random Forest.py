#!/usr/bin/env python
# coding: utf-8

# # Random Forest Classification

# # Importing the libraries
# 

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset
# 

# In[10]:


dataset = pd.read_csv('E:\machineLearning\github\Classification\wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# # Splitting the dataset into the Training set and Test set
# 

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Training the Random Forest Classification model on the Training set
# 

# In[12]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# # Predicting the Test set results
# 

# In[13]:


y_pred = classifier.predict(X_test)


# # RMSE

# In[14]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE : ", rmse)


# # R2 Score

# In[15]:


r2_score = classifier.score(X, y)
print("R2 Score : ", r2_score)

