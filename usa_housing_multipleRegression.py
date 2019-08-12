
# Data Preprocessing Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('USA_Housing.csv')
# X = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
#              'Avg. Area Number of Bedrooms', 'Area Population']]
# y = dataset['Price']
X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 5].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fitting multiple linear regression tree to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict the test set results
y_pred = regressor.predict(X_test)
print(y_test)
print("********************************************************************")
print('************************************************')
print(y_pred)

# using Backward elimination to determine predictors that are statistically important, let the
# significance level be 0.05

import statsmodels.api as sm
X = np.append(arr = np.ones((5000,1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())


