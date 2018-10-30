import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

origin_data = np.loadtxt(open('airfoil_self_noise.dat', 'rb'))   # shape is (1503, 6)

label = origin_data[:, -1]
data = origin_data[:, :-1]  # (1503, 6)
size = len(data)

random_noise_data = np.empty(data.shape, dtype=data.dtype)
random_noise_label = np.empty(label.shape, dtype=label.dtype)
index = np.random.permutation(np.arange(size))

# random shuffle the origin dataset to cross-validation
for i in range(len(index)):
    random_noise_data[i] = data[index[i]]
    random_noise_label[i] = label[index[i]]

k = 5
intervals = int(size/k)
ave_LR_mse, ave_Lasso_mse, ave_Bayesian_mse = 0, 0, 0

for i in range(1, k+1):  # each time in cv
    x_test, y_test = random_noise_data[(i-1)*intervals:i*intervals], random_noise_label[(i-1)*intervals:i*intervals]
    x_train = np.concatenate((random_noise_data[0:(i-1)*intervals], random_noise_data[i*intervals:size]), axis=0)
    y_train = np.concatenate((random_noise_label[0:(i-1)*intervals], random_noise_label[i*intervals:size]), axis=0)

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    diabetes_LR = regr.predict(x_test)
    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    ave_LR_mse += mean_squared_error(y_test, diabetes_LR)
    # # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    regla = linear_model.Lasso()
    regla.fit(x_train, y_train)
    diabetes_Lasso = regla.predict(x_test)
    ave_Lasso_mse += mean_squared_error(y_test, diabetes_Lasso)

    regba = linear_model.BayesianRidge()
    regba.fit(x_train, y_train)
    diabetes_Bayesian = regba.predict(x_test)
    ave_Bayesian_mse += mean_squared_error(y_test,diabetes_Bayesian)

print("LR Mean squared error is:", ave_LR_mse/k)
print("Lasso Mean squared error is:", ave_Lasso_mse/k)
print("Bayesian Regression Mean squared error is:", ave_Bayesian_mse/k)






