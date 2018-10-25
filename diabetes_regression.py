import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


diabetes = datasets.load_diabetes()

diabetes_data = diabetes.data  # (442, 10)
diabetes_label = diabetes.target  # (442,1)

# Use only one feature

# print(diabetes_data[:, np.newaxis, 2:4])
print(diabetes_data)




# Create linear regression object
# regr = linear_model.LinearRegression()
#
# regr.fit(diabetes_X_train, diabetes_y_train)
#
# diabetes_y_pred = regr.predict(diabetes_X_test)

# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
#
# # Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()