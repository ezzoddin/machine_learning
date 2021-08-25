#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classification

# # Importing the libraries
# 

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset
# 

# In[9]:


dataset = pd.read_csv('E:\machineLearning\github\Classification\wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# # Splitting the dataset into the Training set and Test set
# 

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Training the Decision Tree Classification model on the Training set

# In[11]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# # Predicting the Test set results

# In[12]:


y_pred = classifier.predict(X_test)


# # RMSE

# In[13]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE : ", rmse)


# # R2 Score

# In[7]:


r2_score = classifier.score(X, y)
print("R2 Score : ", r2_score)

