# -*- coding: utf-8 -*-


import numpy as np

from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import (linear_kernel,polynomial_kernel,rbf_kernel)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#%% EXAMPLE 1. UNDERSTAND THE KERNEL AND PARAMETERS
#%% Generate sample data
rng = np.random.RandomState(0)

X = 5 * rng.rand(100, 1)
y = np.sin(X).ravel()
# Add noise to targets

yrnd = y + 3 * (0.5 - rng.rand(X.shape[0]))

X_plot = np.linspace(0, 5, 1000)[:, None]

plt.figure(figsize=(8,8))
plt.scatter(X, y, c='b', label='data')
plt.scatter(X, yrnd, c='r', s=10, label='data rnd',zorder=2)
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()

#%% SUBSET GENERATION
# The train, crosvalidation and tes subsets are created
X_train, X_, Y_train, Y_ = train_test_split(X, yrnd, test_size=0.3, random_state=0)
X_cv, X_test, Y_cv, Y_test = train_test_split(X_, Y_, test_size=0.5, random_state=0)

#%% Test a spam of parameter values and calculate the evaluation metric
epsilon = 0.1 # fixed a value

gamma_values = [0.001,0.01,0.1,1,5,10,15,20,25,30, 100] # the values spam to test

mse_train = np.zeros(np.shape(gamma_values))
mse_cv = np.zeros(np.shape(gamma_values))
mse_test = np.zeros(np.shape(gamma_values))


for k in range(len(gamma_values)):
    # K(x, x*) = exp(-gamma ||x-x*||^2)
    model_svr = SVR(kernel='rbf', epsilon=epsilon,gamma=gamma_values[k])
    
    # Step 2. Training the model
    model_svr.fit(X_train,Y_train)

    # Step 3. Using the model
    mse_train[k] = mean_squared_error(Y_train, model_svr.predict(X_train))
    mse_cv[k] = mean_squared_error(Y_cv, model_svr.predict(X_cv))
    mse_test[k] = mean_squared_error(Y_test, model_svr.predict(X_test))


#%% View the performance of the train and cross-validation
fig = plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(gamma_values, mse_train,linewidth=4, markersize=12, c='b',marker='o', label='train')
plt.plot(gamma_values, mse_cv,linewidth=4, markersize=12, c='r',marker='o', label='cv')
plt.plot(gamma_values, mse_test,linewidth=4, markersize=12, c='g',marker='o', label='test')
plt.xlabel('gamma value')
plt.ylabel('MSE')
plt.legend()


plt.subplot(2,1,2)
plt.plot(range(len(gamma_values)), mse_train,linewidth=4, markersize=12, c='b',marker='o', label='train')
plt.plot(range(len(gamma_values)), mse_cv,linewidth=4, markersize=12, c='r',marker='o', label='cv')
plt.plot(range(len(gamma_values)), mse_test,linewidth=4, markersize=12, c='g',marker='o', label='test')
plt.xlabel('index value')
plt.ylabel('MSE')
plt.legend()


#%% Fiting the model selected
epsilon = 0.1

idx=int(np.argwhere(mse_train == min(mse_train))[0])
idx=int(np.argwhere(mse_cv == min(mse_cv))[0])
idx=int(np.argwhere(mse_test == min(mse_test))[0])


# K(x, x*) = exp(-gamma ||x-x*||^2)
model_svr_best = SVR(kernel='rbf', epsilon=epsilon,gamma=gamma_values[idx])

# Step 2. Training the model
model_svr_best.fit(X,yrnd)

# Step 3. Using the model
y_hat = model_svr_best.predict(X)

# Step 4. Evaluation of results
sv_x = model_svr_best.support_
Y_plot = model_svr_best.predict(X_plot)


#%% View the performance of the best model
# #############################################################################
# View the results
fig = plt.figure(figsize=(30,15))
plt.subplot(1,2,1)
plt.scatter(X, y, c='b', label='data')
plt.scatter(X, yrnd, c='r', s=10, label='data rnd',zorder=2)
plt.scatter(X[sv_x], yrnd[sv_x], c='k',s=60, marker='x', label='SVR support vectors', zorder=1,edgecolors=(0, 0, 0))
plt.plot(X_plot, Y_plot, c='k',label='SVR regression')
plt.plot(X_plot, Y_plot+epsilon, c='k', linestyle='dashed',label='SVR+\epsilon')
plt.plot(X_plot, Y_plot-epsilon, c='k', linestyle='dashed',label='SVR-\epsilon')
plt.xlabel('data')
plt.ylabel('target')
plt.title('R^2 = %0.4f'%model_svr.score(X,yrnd))
plt.legend()

plt.subplot(1,2,2)
plt.scatter(y_hat, yrnd, c='b', label='Estimation')
plt.plot(yrnd, yrnd, c='k', label='Perfect estimation')
plt.xlabel('Y estimated')
plt.ylabel('Y real')
plt.title('R^2 = %0.4f'%model_svr.score(X,yrnd))
plt.legend()
plt.grid()
plt.show()




#%% Import the model for later
#% Save the model trained
import pickle
pickle.dump(model_svr,open('model.sav','wb'))

#% Load the model saved
model_svr_loaded = pickle.load(open('model.sav','rb'))