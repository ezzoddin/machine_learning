import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score


dataset = pd.read_excel(r'E:\machineLearning\github\CCPP\CCPP.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#nb_dgree = 2
#nb_dgree = 3
#nb_dgree = 4
nb_dgree = 5
polynomial_features = PolynomialFeatures(degree = nb_dgree)
X_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)


mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y,y_pred)

print('RMSE: ', rmse)
print('R2: ', r2)



