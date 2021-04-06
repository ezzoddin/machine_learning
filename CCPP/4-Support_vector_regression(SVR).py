#!/usr/bin/env python
# coding: utf-8

# # Support Vector Regression (SVR)

# In[10]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

dataset = pd.read_excel(r'E:\machineLearning\github\CCPP\CCPP.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

xtrain , xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2, random_state=42)

svr_kernel = 'rbf'
#svr_kernel = 'linear'
#svr_kernel = 'Sigmoid'
#svr_kernel = 'poly'
#svr_dgree = 2
svr_dgree = 3
#svr_dgree = 4
#svr_dgree = 5
model = SVR(kernel = svr_kernel, degree = svr_dgree)

model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

mse = mean_squared_error(ytest, ypred)
rmse = np.sqrt(mse)
r2_score = model.score(X, y)

print("RMSE : ", rmse)
print("R2 Score : ", r2_score)


# In[ ]:




