# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:39:29 2024

@author: zaratejo
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
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
results_df = pd.DataFrame(columns=['Model', 'MSE_Train', 'MSE_Test', 'MSE_CV', 'R2_Train', 'R2_CV', 'R2_Test', 'kernel', 'C', 'epsilon', 'gamma','Train_Time'])

 


#%% Define the parameter grid for SVR
param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.5, 1],
    'gamma': ['scale', 'auto']  # You can also include numerical values if needed
}

# Initialize the SVR model
svr = SVR()


#%% Define the parameter grid for SVR
# Function to perform grid search manually using the custom CV split
def manual_grid_search(model, param_grid, X_train, Y_train, X_test, Y_test, X_cv, Y_cv):
    best_params = None
    best_score = float('inf')
    best_model = None
    all_results = []

    for kernel in param_grid['kernel']:
        for C in param_grid['C']:
            for epsilon in param_grid['epsilon']:
                for gamma in param_grid['gamma']:
                    # Set parameters for the model
                    model.set_params(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

                    # Start timing
                    start_time = time.time()

                    # Train the model
                    model.fit(X_train, Y_train)

                    # End timing
                    train_time = time.time() - start_time

                    # Evaluate on training, cross-validation, and test sets
                    y_pred_train = model.predict(X_train)
                    y_pred_cv = model.predict(X_cv)
                    y_pred_test = model.predict(X_test)

                    mse_train = mean_squared_error(Y_train, y_pred_train)
                    mse_cv = mean_squared_error(Y_cv, y_pred_cv)
                    mse_test = mean_squared_error(Y_test, y_pred_test)

                    r2_train = r2_score(Y_train, y_pred_train)
                    r2_test = r2_score(Y_test, y_pred_test)
                    r2_cv = r2_score(Y_cv, y_pred_cv)

                    # Save results for this parameter set
                    all_results.append({
                        'Model': 'SVR',
                        'kernel': kernel,
                        'C': C,
                        'epsilon': epsilon,
                        'gamma': gamma,
                        'MSE_Train': mse_train,  # Store training MSE
                        'MSE_CV': mse_cv,        # Store cross-validation MSE
                        'MSE_Test': mse_test,    # Store test MSE
                        'R2_Train': r2_train,    # Store training R²
                        'R2_CV': r2_cv,          # Store cross-validation R²
                        'R2_Test': r2_test,    # Store training R²
                        'Train_Time': train_time  # Store training time
                    })

                    # Check if this is the best model so far
                    if mse_cv < best_score:
                        best_score = mse_cv
                        best_params = {'kernel': kernel, 'C': C, 'epsilon': epsilon, 'gamma': gamma}
                        best_model = model

    return best_model, best_params, best_score, all_results


# Function to convert results into a DataFrame
def display_results(results):
    return pd.DataFrame(results)

# Perform manual grid search using the custom split for SVR
best_svr, best_params_svr, best_score_svr, svr_results = manual_grid_search(svr, param_grid_svr, X_train, Y_train,  X_test,Y_test,X_cv, Y_cv)

# Convert the results to a DataFrame
svr_results_df = display_results(svr_results)

svr_results_df_best = svr_results_df.nsmallest(1, 'MSE_CV')

# Concatenate with results_df
results_df = pd.concat([results_df, svr_results_df_best])


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
   
    'MSE_Train': mse_train_lr,  # Store training MSE
    'MSE_CV': mse_cv_lr,         # Store cross-validation MSE
    'MSE_Test': mse_test_lr,     # Store test MSE
    'R2_Train': r2_score(Y_train, y_pred_train_lr),  # Store training R²
    'R2_CV': r2_score(Y_cv, y_pred_cv_lr),           # Store cross-validation R²
    'R2_Test': r2_score(Y_test, y_pred_test_lr),     # Store test R²
    'Train_Time': train_time_lr  # Store training time
}

# Append the new row for Linear Regression to the results DataFrame
results_df = pd.concat([results_df, pd.DataFrame([new_row_lr])])#, ignore_index=True)

