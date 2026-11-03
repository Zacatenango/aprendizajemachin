# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:49:36 2023

@author: zaratejo
"""

import tensorflow as tf
from keras.models import Sequential   # use sequential to design network
from tensorflow.keras.models import Model # more complex networks, less stable
import keras 
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import tensorflow as tf
import pandas as pd
from keras.layers import Input, Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.datasets as data
from ucimlrepo import fetch_ucirepo 
import seaborn as sns
 
from ucimlrepo import fetch_ucirepo 
wine_quality = fetch_ucirepo(id=186) 
data=wine_quality.data.original




#%% DATA QUALITY REPORT
def dqr(data):
    
    cols = pd.DataFrame(list(data.columns.values),
                           columns=['Name'],
                           index=list(data.columns.values))
    dtyp = pd.DataFrame(data.dtypes,columns=['Type'])
    misval = pd.DataFrame(data.isnull().sum(),
                                  columns=['N/A value'])
    presval = pd.DataFrame(data.count(),
                                  columns=['Count values'])
    unival = pd.DataFrame(columns=['Unique values'])
    minval = pd.DataFrame(columns=['Min'])
    maxval = pd.DataFrame(columns=['Max'])
    mean =pd.DataFrame(data.mean(), columns=['Mean']) 
    Std =pd.DataFrame(data.std(), columns=['Std']) 
    Var =pd.DataFrame(data.var(), columns=['Var']) 
    median =pd.DataFrame(data.median(), columns=['Median']) 
    
    skewness = pd.DataFrame(data.skew(), columns=['Skewness']) 
    kurtosis = pd.DataFrame(data.kurtosis(), columns=['Kurtosis']) 

    for col in list(data.columns.values):
        unival.loc[col] = [data[col].nunique()]
        try:
            minval.loc[col] = [data[col].min()]
            maxval.loc[col] = [data[col].max()]
        except:
            pass
    
    # Juntar todas las tablas
    return cols.join(dtyp).join(misval).join(presval).join(unival).join(minval).join(maxval).join(mean).join(Std).join(Var).join(median).join(skewness).join(kurtosis)

#%% Obtaining the data quality report
data['color']=data.color.replace("red",1).replace("white",0)
report = dqr(data)
#print(data.info())

#%%

X = wine_quality.data.features 
Y=data[['quality', 'color']]
#Y['color']=Y.color.replace("red",1).replace("white",0)

# Spliting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


#%%


# Input layer
input_layer = Input(shape=(11,), name='input_layer')

# Shared hidden layers
shared_layer = Dense(7, activation='tanh')(input_layer)
shared_layer = Dense(7, activation='sigmoid')(shared_layer)
#shared_layer = Dense(10, activation='tanh')(shared_layer)

# Output for quality (regression)
output_quality = Dense(1, activation='linear', name='quality_output')(shared_layer)

# Output for color (classification)
output_color = Dense(1, activation='sigmoid', name='color_output')(shared_layer)

# Create model
model = Model(inputs=input_layer, outputs=[output_quality, output_color])

model.summary()

#%% View the model
from keras.utils import plot_model
plot_model(model)
# plot_model(model, to_file='../figures/P9_fig/model.png', show_shapes=True)

plot_model(model, show_shapes=True, show_layer_names=True)

#%% How to train the NN?

Y_train_quality = Y_train['quality']
Y_train_color = Y_train['color']
Y_test_quality = Y_test['quality']
Y_test_color = Y_test['color']

#%% Optimizer
epochs = 200
learning_rate = 0.1
decay_rate = learning_rate/epochs
momentum = 0.8

opt = keras.optimizers.SGD(learning_rate=learning_rate,
                            weight_decay=decay_rate,
                            nesterov=True)
opt = keras.optimizers.Adam(learning_rate=learning_rate)

#%% Configure
model.compile(optimizer=opt,
              loss={'quality_output': 'mse', 'color_output': 'binary_crossentropy'},
              metrics={'quality_output': 'mse', 'color_output': 'accuracy'}) #or mae



#%% Train the model
history = model.fit(X_train,
                    {'quality_output': Y_train_quality, 'color_output': Y_train_color},
                    epochs=100, batch_size=32,
                    validation_data=(X_test, {'quality_output': Y_test_quality, 'color_output': Y_test_color}))

# Evaluate the model performance on test data
loss, quality_loss, color_loss, quality_mae, color_accuracy = model.evaluate(X_test,
                                                                             {'quality_output': Y_test_quality,
                                                                              'color_output': Y_test_color})
print(f"Quality MAE: {quality_mae}, Color Accuracy: {color_accuracy}")

# Plot the evolution of the cost function for both classification and regression
def plot_loss(history):
    # Extract loss values for training and validation
    loss_quality = history.history['quality_output_loss']
    val_loss_quality = history.history['val_quality_output_loss']
    loss_color = history.history['color_output_loss']
    val_loss_color = history.history['val_color_output_loss']

    # Plotting
    epochs = range(1, len(loss_quality) + 1)
    plt.figure(figsize=(14, 6))

    # Plot for regression loss (Quality)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_quality, label='Training Quality Loss (MSE)')
    plt.plot(epochs, val_loss_quality, label='Validation Quality Loss (MSE)')
    plt.title('Quality Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    # Plot for classification loss (Color)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_color, label='Training Color Loss (Binary Crossentropy)')
    plt.plot(epochs, val_loss_color, label='Validation Color Loss (Binary Crossentropy)')
    plt.title('Color Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Binary Crossentropy)')
    plt.legend()

    plt.show()

# Call the plotting function
plot_loss(history)



#%%

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

#%%
# Get predictions for the test and training sets
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Extract the predictions for quality and color separately
train_quality_pred = train_predictions[0].flatten()  # Regression output (quality)
train_color_pred = (train_predictions[1] > 0.5).astype(int).flatten()  # Classification output (color)

test_quality_pred = test_predictions[0].flatten()  # Regression output (quality)
test_color_pred = (test_predictions[1] > 0.5).astype(int).flatten()  # Classification output (color)
#%%
# Calculate metrics for the training set
train_accuracy_color = accuracy_score(Y_train_color, train_color_pred)
train_mse_quality = mean_squared_error(Y_train_quality, train_quality_pred)
train_r2_quality = r2_score(Y_train_quality, train_quality_pred)

# Calculate metrics for the test set
test_accuracy_color = accuracy_score(Y_test_color, test_color_pred)
test_mse_quality = mean_squared_error(Y_test_quality, test_quality_pred)
test_r2_quality = r2_score(Y_test_quality, test_quality_pred)

# Print training metrics
print("Training Metrics:")
print(f"Color Classification Accuracy: {train_accuracy_color:.4f}")
print(f"Quality Regression MSE: {train_mse_quality:.4f}")
print(f"Quality Regression R²: {train_r2_quality:.4f}")

# Print testing metrics
print("\nTesting Metrics:")
print(f"Color Classification Accuracy: {test_accuracy_color:.4f}")
print(f"Quality Regression MSE: {test_mse_quality:.4f}")
print(f"Quality Regression R²: {test_r2_quality:.4f}")

