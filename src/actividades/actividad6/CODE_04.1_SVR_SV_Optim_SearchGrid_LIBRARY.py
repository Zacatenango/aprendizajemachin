# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:39:29 2024

@author: zaratejo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import time
from sklearn.linear_model import LinearRegression

#%% Generate sample data (regression task)
rng = np.random.RandomState(0)
X = 5 * rng.rand(100, 1)
y = np.sin(X).ravel()
# Add noise to targets
yrnd = y + 3 * (0.5 - rng.rand(X.shape[0]))

#%% Visualize the data
plt.figure(figsize=(8, 8))
plt.scatter(X, y, c='b', label='data')
plt.scatter(X, yrnd, c='r', s=10, label='data with noise', zorder=2)
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()

#%% SUBSET GENERATION
# Split the data into training, cross-validation, and test sets
X_train, X_, Y_train, Y_ = train_test_split(X, yrnd, test_size=0.3, random_state=0)
X_cv, X_test, Y_cv, Y_test = train_test_split(X_, Y_, test_size=0.5, random_state=0)

#%% Initialize DataFrame for results
results_df = pd.DataFrame(columns=['Model', 'MSE_Train', 'MSE_Test', 'MSE_CV', 'MSE_Mul_CV', 
                                   'R2_Train', 'R2_CV', 'R2_Test', 'R2_Mul_CV', 
                                   'kernel', 'C', 'epsilon', 'gamma', 'Train_Time'])

#%% Define the parameter grid for SVR
param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.5, 1],
    'gamma': ['scale', 'auto']  # You can also include numerical values if needed
}

# Initialize the SVR model
svr = SVR()

#%% Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid_svr, 
                           scoring='neg_mean_squared_error', cv=5, 
                         verbose=1,
                            refit=True,
                           return_train_score=True, n_jobs=-1)

# Start timing for the grid search
start_time = time.time()

# Fit the model
grid_search.fit(X_train, Y_train)

# End timing
train_time = time.time() - start_time

# Get the best model and parameters
best_svr = grid_search.best_estimator_
best_params_svr = grid_search.best_params_
best_score_svr = -grid_search.best_score_  # Convert back from negative MSE
Results=grid_search.cv_results_
print(Results)
# Predict and evaluate
y_pred_train = best_svr.predict(X_train)
y_pred_cv = best_svr.predict(X_cv)
y_pred_test = best_svr.predict(X_test)

# Calculate metrics
mse_train = mean_squared_error(Y_train, y_pred_train)
mse_cv = mean_squared_error(Y_cv, y_pred_cv)
mse_test = mean_squared_error(Y_test, y_pred_test)

r2_train = r2_score(Y_train, y_pred_train)
r2_cv = r2_score(Y_cv, y_pred_cv)
r2_test = r2_score(Y_test, y_pred_test)

# Create a DataFrame for the results
svr_results = {
    'Model': 'SVR',
    'kernel': best_params_svr['kernel'],
    'C': best_params_svr['C'],
    'epsilon': best_params_svr['epsilon'],
    'gamma': best_params_svr['gamma'],
    'MSE_Train': mse_train,
    'MSE_CV': mse_cv,
    'MSE_Test': mse_test,
    'R2_Train': r2_train,
    'R2_CV': r2_cv,
    'R2_Test': r2_test,
    'Train_Time': train_time
}

# Append the results to results_df
results_df = pd.concat([results_df, pd.DataFrame([svr_results])], ignore_index=True)

#%% Linear Regression Model
linear_model = LinearRegression()

# Start timing for Linear Regression
start_time = time.time()

# Train the Linear Regression model
linear_model.fit(X_train, Y_train)

# End timing
train_time_lr = time.time() - start_time

# Evaluate on training, cross-validation, and test sets
y_pred_train_lr = linear_model.predict(X_train)
y_pred_cv_lr = linear_model.predict(X_cv)
y_pred_test_lr = linear_model.predict(X_test)

mse_train_lr = mean_squared_error(Y_train, y_pred_train_lr)  # Calculate training MSE
mse_cv_lr = mean_squared_error(Y_cv, y_pred_cv_lr)
mse_test_lr = mean_squared_error(Y_test, y_pred_test_lr)

# Store results for Linear Regression in the DataFrame
new_row_lr = {
    'Model': 'Linear Regression',
    'MSE_Train': mse_train_lr,
    'MSE_CV': mse_cv_lr,
    'MSE_Test': mse_test_lr,
    'R2_Train': r2_score(Y_train, y_pred_train_lr),
    'R2_CV': r2_score(Y_cv, y_pred_cv_lr),
    'R2_Test': r2_score(Y_test, y_pred_test_lr),
    'Train_Time': train_time_lr
}

# Append the new row for Linear Regression to the results DataFrame
results_df = pd.concat([results_df, pd.DataFrame([new_row_lr])], ignore_index=True)

# Optionally display the final results DataFrame
print(results_df)
