#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine (SVM)
# 

# # Importing the libraries
# 

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset
# 

# In[5]:


dataset = pd.read_csv('E:\machineLearning\github\Classification\wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# # Splitting the dataset into the Training set and Test set
# 

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Training the SVM model on the Training set
# 

# In[7]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# # Predicting the Test set results

# In[9]:


y_pred = classifier.predict(X_test)


# # RMSE

# In[10]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE : ", rmse)


# # R2 Score

# In[11]:


r2_score = classifier.score(X, y)
print("R2 Score : ", r2_score)

