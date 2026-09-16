# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:15:20 2024

@author: zaratejo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% IMPORT THE DATA SET
data = pd.read_csv('glass.data',header=None)
names = ['ID','Refractive_index','Na','Mg','Al','Si','K',
         'Ca','Ba','Fe','Glass_type']
data.columns = names

# Remove the 'ID' and 'Glass_type' columns
data = data.drop(['ID', 'Glass_type'], axis=1)

# Separate the target variable and the features
X = data.drop('Refractive_index', axis=1)
Y = data['Refractive_index']
#%%
# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)




# Make predictions on the test set
y_pred = regressor.predict(X_train)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
print('Mean squared error: %.10f' % mean_squared_error(y_train, y_pred))
print('R2: %.2f' % r2_score(y_train, y_pred))


import numpy as np

def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

mse = mean_squared_error(y_train, y_pred)
print('Mean squared error: %.10f' % mse)



#%%
import matplotlib.pyplot as plt


# Plot the actual vs predicted values
plt.scatter( y_pred,y_train)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LINEAR Actual vs Predicted values')
min_val = min(Y.min(), y_pred.min())
max_val = max(Y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r')

plt.show()

#%%
# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
print('Mean squared error: %.10f' % mean_squared_error(y_test, y_pred))
print('R2: %.2f' % r2_score(y_test, y_pred))

#%%
import matplotlib.pyplot as plt


# Plot the actual vs predicted values
plt.scatter( y_pred,y_test)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LINEAR Actual vs Predicted values')
min_val = min(Y.min(), y_pred.min())
max_val = max(Y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r')

plt.show()


#%%

import statsmodels.api as sm
# Fit the linear regression model using statsmodels
model = sm.OLS(Y, sm.add_constant(X)).fit() 

# Predict the values
y_pred = model.predict(sm.add_constant(X))

# Print metrics
print('Mean squared error: %.10f' % mean_squared_error(Y , y_pred))
print('R2: %.2f' % r2_score(Y , y_pred))
print(model.summary())
#%%
# Train a Ridge regression model ridge regression add a constraint on the coefficients 
#Ridge regression is a type of linear regression that uses L2 regularization to prevent overfitting.
#the square values of the coefficients multiplied by a regularization parameter. 
y_pred = regressor.predict(X)

from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X, Y)

# Make predictions on the test set
y_pred_ridge = ridge.predict(X)

# Evaluate the model
print('Ridge Regression')
print('Mean squared error: %.10f' % mean_squared_error(Y, y_pred_ridge))
print('R2: %.2f' % r2_score(Y, y_pred_ridge))

# Train a Lasso regression model
#Lasso regression is a type of linear regression that uses L1 regularization to prevent overfitting
#the absolute values of the coefficients multiplied by a regularization parameter. 
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X, Y)

# Make predictions on the test set
y_pred_lasso = lasso.predict(X)

# Evaluate the model
print('Lasso Regression')
print('Mean squared error: %.10f' % mean_squared_error(Y, y_pred_lasso))
print('R2: %.2f' % r2_score(Y, y_pred_lasso))



plt.figure(figsize=(10, 8))

# Scatter plots
plt.scatter(y_pred, Y, label='Linear', color='blue', alpha=0.6)
plt.scatter(y_pred_ridge, Y, label='Ridge', color='orange', alpha=0.6)
plt.scatter(y_pred_lasso, Y, label='Lasso', color='green', alpha=0.6)



# Reference line
min_val = min(Y.min(), y_pred.min(), y_pred_ridge.min(), y_pred_lasso.min())
max_val = max(Y.max(), y_pred.max(), y_pred_ridge.max(), y_pred_lasso.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True, alpha=0.3)



plt.show()

