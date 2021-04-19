import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import train_test_split

dataset = pd.read_excel(r'E:\machineLearning\github\CCPP\CCPP.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

xtrain , xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

mse = mean_squared_error(ytest, ypred)
rmse = np.sqrt(mse)
r2_score = model.score(X, y)

print("RMSE : ", rmse)
print("R2 Score : ", r2_score)