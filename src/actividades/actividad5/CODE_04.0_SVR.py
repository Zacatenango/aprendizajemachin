# -*- coding: utf-8 -*-


import numpy as np

from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import (linear_kernel,polynomial_kernel,rbf_kernel)

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
plt.scatter(X, yrnd, c='r', s=10, label='data rnd')#,zorder=2)
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()

#%% Kernel transformation
# K_x =  linear_kernel(X) # K(x, x*) = <x, x*>
# K_x =  polynomial_kernel(X,degree=2,coef0=5) # K(x, x*) = (gamma <x, x*> + coef0)^degree
# K_x =  rbf_kernel(X,gamma=0.01) # K(x, x*) = exp(-gamma ||x-x*||^2)
#%%


#%% Training a SVR model
epsilon = 0.1

# Step 1. Create the model
# K(x, x*) = <x, x*>
#model_svr = SVR(kernel='linear', epsilon=epsilon)

#K(x, x*) = (gamma <x, x*> + coef0)^degree
model_svr = SVR(kernel='poly', epsilon=epsilon, degree=8,coef0=0)

# K(x, x*) = exp(-gamma ||x-x*||^2)
#model_svr = SVR(kernel='rbf', epsilon=epsilon,gamma=1)

#radial basis function
## K(x,x*) = tanh(gamma <x, x*> + coef0)
#model_svr = SVR(kernel='sigmoid', epsilon=epsilon,gamma=10,coef0=1)


# Step 2. Training the model
model_svr.fit(X,yrnd)

# Step 3. Using the model
y_hat = model_svr.predict(X)

# Step 4. Evaluation of results
sv_x = model_svr.support_

R2 = model_svr.score(X,yrnd)

Y_plot = model_svr.predict(X_plot)

#%% Prediction using the optimization problem results.
alphas = model_svr.dual_coef_
x_sv = model_svr.support_vectors_
b = model_svr.intercept_

#K_x = linear_kernel(x_sv,X_plot) # Needs the correct Kernel
#Y_p = np.dot(alphas,K_x)+b # <W,X>+b

#%%
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
plt.savefig('SVR.pdf')  
plt.show()



#%% EXAMPLE 2. ROBUSTNESS REVIEW
#%% Generate sample data
rng = np.random.RandomState(0)

# #############################################################################
X = 5 * rng.rand(100, 1)
y = np.ravel(3*X+2)

# Add noise to targets
yrnd = y + 3 * (0.5 - rng.rand(X.shape[0]))
yrnd[::20] += 50 * (0.5 - rng.rand(X.shape[0]//20))

X_plot = np.linspace(0, 5, len(X))[:, None]


plt.figure(figsize=(8,8))
plt.scatter(X, y, c='b', label='data')
plt.scatter(X, yrnd, c='r', s=10, label='data rnd',zorder=2)
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()

#%% Create a linear regressor to compare
from sklearn.linear_model import LinearRegression
modelo_lin = LinearRegression().fit(X,yrnd)
Ylin_plot = modelo_lin.predict(X_plot)

plt.figure(figsize=(8,8))
plt.scatter(X, y, c='b', s=10, label='data')
plt.scatter(X, yrnd, c='r', s=10, label='data rnd',zorder=2)
plt.scatter(X_plot, Ylin_plot, c='g', s=10, label='linear',zorder=2)
plt.xlabel('input',fontsize=15)
plt.ylabel('target',fontsize=15)
plt.legend(fontsize=15)
plt.show()

#%% Create a SVR model
epsilon = 0.1
model_svr =SVR(kernel='linear', epsilon=epsilon)
model_svr.fit(X,yrnd)
Ysvr_plot = model_svr.predict(X_plot)

plt.figure(figsize=(8,8))
plt.scatter(X, y, c='b', s=10, label='data')
plt.scatter(X, yrnd, c='r', s=10, label='data rnd',zorder=2)
plt.scatter(X_plot, Ylin_plot, c='g', s=10, label='lineal',zorder=2)
plt.scatter(X_plot, Ysvr_plot, c='m', s=10, label='svr',zorder=2)
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()


#%% Import the model for later
#% Save the model trained
import pickle
pickle.dump(model_svr,open('model.sav','wb'))

#% Load the model saved
model_svr_loaded = pickle.load(open('model.sav','rb'))