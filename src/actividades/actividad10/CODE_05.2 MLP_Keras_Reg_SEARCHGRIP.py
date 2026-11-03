# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:54:29 2025

@author: zaratejo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:11:26 2023

@author: zaratejo
"""
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# -*- coding: utf-8 -*-

#pip install keras
#pip install tensorflow

#pip install pydot
#pip install graphviz

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV




#%% Import dataset
data = pd.read_csv('airfoil_self_noise.dat', sep='\t', header=None)
names = ['Freq', 'Angle', 'Chord', 'Velocity', 'Suction', 'Sound']
data.columns = names

# Data input and output selection
X = data[names[0:5]]
Y = data[names[5]]

# Scaling dataset
X = MinMaxScaler().fit_transform(X)
Y = MinMaxScaler().fit_transform(np.resize(Y, (1503, 1)))

#%% Dataset split (Training and Test)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3)

#%% Define the model function
def create_model(learning_rate=0.01, momentum=0.0):
    model = Sequential([
        tf.keras.Input(shape=(5,), name='input_layer'),
        Dense(5, activation='tanh'),
        Dropout(0.2),
        Dense(3, activation='tanh'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    return model

# Wrap the model
model = KerasRegressor(model=create_model, epochs=10, batch_size=200, verbose=0,
                       validation_split=0.2)  # needed so history has val_loss / val_mse

# Define the grid search parameters
QUICK_TEST = False  # set to False for the full search

if QUICK_TEST:
    # 2 combinations x 2 folds = 4 short trainings
    param_grid = {
        'model__learning_rate': [0.01, 0.1],
        'model__momentum': [0.9],
        'epochs': [10],
        'batch_size': [200]
    }
    cv_folds = 2
else:
    # 81 combinations x 3 folds = 243 trainings
    param_grid = {
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__momentum': [0.0, 0.5, 0.9],
        'epochs': [10, 50, 100],
        'batch_size': [50, 100, 200]
    }
    cv_folds = 3

# Perform grid search (n_jobs=-1 uses all CPU cores)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error',
                    cv=cv_folds, n_jobs=-1)
grid_result = grid.fit(xtrain, ytrain)

# Print the best parameters
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print(f"{mean} ({std}) with: {param}")

#%% Performance evaluation with the best model
best_model = grid.best_estimator_.model_
score = best_model.evaluate(xtest, ytest)
print('Test loss:', score[0])
print('Test mse:', score[1])

#%% Plot the loss function and the metrics
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(grid_result.best_estimator_.model_.history.history['loss'], 'r', label='train')
ax.plot(grid_result.best_estimator_.model_.history.history['val_loss'], 'b', label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

# Plot the mse
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(np.sqrt(grid_result.best_estimator_.model_.history.history['mse']), 'r', label='train')
ax.plot(np.sqrt(grid_result.best_estimator_.model_.history.history['val_mse']), 'b', label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'MSE', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

#%% Predictions and R^2 score
yhat = best_model.predict(xtrain)
R2_score = r2_score(ytrain, yhat)

xmin, xmax = min(ytrain), max(ytrain)
xline = np.linspace(xmin, xmax)
fig = plt.figure(figsize=(10, 6))
plt.scatter(ytrain, yhat, label='Estimation')
plt.plot(xline, xline, 'k--', label='Perfect estimation')
plt.xlabel('Real output', fontsize=20)
plt.ylabel('Estimation output', fontsize=20)
plt.title('R^2=%0.4f' % R2_score, fontsize=20)
plt.legend()
plt.grid()
plt.show()

