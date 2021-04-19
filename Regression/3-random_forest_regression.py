import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_excel(r'E:\machineLearning\github\CCPP\CCPP.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#nb_estimators = 5
#nb_estimators = 10
#nb_estimators = 20
#nb_estimators = 50
#nb_estimators = 100
#nb_estimators = 200
#nb_estimators = 500
nb_estimators = 1000


model = RandomForestRegressor(n_estimators = nb_estimators , random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_score = model.score(X, y)

print("RMSE : ", rmse)
print("R2 Score : ", r2_score)

