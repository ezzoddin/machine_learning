import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
  
dataset = pd.read_excel(r'E:\machineLearning\github\CCPP\CCPP.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

train_X, test_X, train_y, test_y = train_test_split(X, y,test_size = 0.3, random_state = 123)
  
xgb_r = xg.XGBRegressor(objective ='reg:linear',n_estimators = 10, seed = 123)
xgb_r.fit(train_X, train_y)

y_pred = xgb_r.predict(test_X)

mse = mean_squared_error(test_y, y_pred)
rmse = np.sqrt(mse)

print("RMSE : ", np.round(rmse,3))

