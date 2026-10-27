# -*- coding: utf-8 -*-
"""

"""
# Old versions
#conda install -c conda-forge keras
#conda install -c conda-forge/label/broken keras
#conda install -c conda-forge/label/cf201901 keras
#conda install -c conda-forge tensorflow=1.14

# Lastest version
#pip install keras
#pip install tensorflow

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate dummy data
import numpy as np
from sklearn.metrics import r2_score
import seaborn as sns  # for the heatmap


#%% UNIVARIATE EXAMPLE: Generation of data in 1D
m = 1000;
X = np.linspace(-1,1,num=m)
X = np.reshape(X,(m,1))

Y = X**2 # Y = X^21
Y = 2*X**3+2*X**2 # Y = 2X^3+2X^2
Y = 4*X*np.sin(10*X)+X*np.exp(X)*np.cos(20*X) # Y = 4X*sin(10X)+X*exp^(X)*cos(20X)

[m,ne] = np.shape(X)

from sklearn import preprocessing
X = preprocessing.scale(X)
Y = preprocessing.scale(Y)

fig = plt.figure()
plt.plot(X,Y, linewidth=3)
plt.grid()

#%% Creation of the neural network for the 1D problem

# https://keras.io/getting-started/sequential-model-guide/
model = Sequential(
    [tf.keras.Input(shape = (1,),name='input_layer'),
    Dense(10, activation='linear',name='hidden_layer1'),
    #Dense(10, activation='tanh',name='hidden_layer2'),
    #Dense(10, activation='tanh',name='hidden_layer3'),
    Dense(10, activation='relu',name='hidden_layer4'),
    Dense(10, activation='relu',name='hidden_layer5'),
    #Dense(500, activation='tanh',name='hidden_layer6'),
    Dense(1, activation='linear',name='output_layer')
    ], name = 'my_model'
    )

model = Sequential(
    [tf.keras.Input(shape = (1,),name='input_layer'),
    Dense(2, activation='linear',name='hidden_layer1'),
    Dense(30, activation='relu',name='hidden_layer2'),
    Dense(30, activation='linear',name='hidden_layer3'),
    Dense(30, activation='relu',name='hidden_layer4'),
    #Dense(5, activation='tanh',name='hidden_layer5'),
    #Dense(10, activation='tanh',name='hidden_layer6'),
    Dense(1, activation='linear',name='output_layer')
    ], name = 'my_model'
    )

model = Sequential(
    [tf.keras.Input(shape = (1,),name='input_layer'),
    Dense(10, activation='linear',name='hidden_layer1'),
    Dense(5, activation='tanh',name='hidden_layer2'),
    Dense(5, activation='tanh',name='hidden_layer3'),
    Dense(10, activation='relu',name='hidden_layer4'), #Ayuda la ReLu para que mejorara el modelo
    Dense(5, activation='tanh',name='hidden_layer5'),
    Dense(5, activation='tanh',name='hidden_layer6'),
    Dense(1, activation='linear',name='output_layer') #Es una capa como las ocultas y no sabe que es la última capa
    ], name = 'my_model'
    )

model = Sequential(
    [tf.keras.Input(shape = (1,),name='input_layer'),
    Dense(10, activation='relu',name='hidden_layer4'),
    Dense(10, activation='relu',name='hidden_layer5'),
    Dense(10, activation='relu',name='hidden_layer6'),
    Dense(1, activation='linear',name='output_layer')
    ], name = 'my_model'
    )                      
# https://keras.io/activations/

# https://keras.io/optimizers/
sgd = SGD(learning_rate=0.001, weight_decay=0.001, momentum=0.99)
adam = Adam()

#https://keras.io/api/losses/
model.compile(loss='mean_squared_error',
              optimizer=adam)

history = model.fit(X, Y,  epochs=5, batch_size=100,verbose=1)

score = model.evaluate(X, Y)

Yhat = model.predict(X)


J = history.history['loss']
R2_score = r2_score(Y,Yhat)


# Display of the estimate by the regression
fig = plt.figure(figsize=(15,8))
plt.plot(X[:,0],Y,label='Y real', linewidth=3)
plt.plot(X[:,0],Yhat,label='Y estimated', linewidth=3)
plt.title('Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

#%%
import itertools

learning_rates = [0.001, 0.01, 0.1]
weight_decays = [0.001, 0.01, 0.1]
momentums = [0.5, 0.9, 0.99]

best_r2 = -float('inf')
best_params = {}

for lr, wd, mom in itertools.product(learning_rates, weight_decays, momentums):
    sgd = SGD(learning_rate=lr, weight_decay=wd, momentum=mom)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    history = model.fit(X, Y, epochs=300, verbose=0, batch_size=1000)
    Yhat = model.predict(X)
    r2 = r2_score(Y, Yhat)
    if r2 > best_r2:
        best_r2 = r2
        best_params = {'learning_rate': lr, 'weight_decay': wd, 'momentum': mom}

print("Best R2 score:", best_r2)
print("Best parameters:", best_params)

#%% Visualize the evolution of learning
fig = plt.figure(figsize=(15,8))
plt.plot(np.arange(len(J)),J, linewidth=3)
plt.title('J(W) vs epoch: J(W)=%0.4f'%J[-1])
plt.xlabel('epoch')
plt.ylabel('J(W)')
plt.grid()
plt.show()


# Display of the estimate by the regression
fig = plt.figure(figsize=(15,8))
plt.plot(X[:,0],Y,label='Y real', linewidth=3)
plt.plot(X[:,0],Yhat,label='Y estimated', linewidth=3)
plt.title('Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

fig = plt.figure(figsize=(15,8))
plt.scatter(Y,Yhat)
plt.plot(Y,Y,'k--',linewidth=3)
plt.axis('square')
plt.title('Prediction: R^2=%0.4f'%R2_score)
plt.xlabel('Y')
plt.ylabel('Yhat')
plt.grid()
plt.show()

#%% Heatmap of the weights in the hidden layers

# Get weights from the model
weights = model.get_weights()

# Visualize the weights of each layer
for i, layer_weights in enumerate(weights):
    if len(layer_weights.shape) == 2:  # Only consider the weight matrices, not biases
        fig = plt.figure(figsize=(8,6))
        sns.heatmap(layer_weights, annot=True, cmap='coolwarm', cbar=True)
        plt.title(f'Heatmap of Weights for Layer {i//2 + 1}')
        plt.xlabel('Neurons')
        plt.ylabel('Input Features' if i == 0 else 'Neurons from Previous Layer')
        plt.show()
