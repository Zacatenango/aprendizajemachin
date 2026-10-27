# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:00:12 2023

@author: zaratejo
"""

# -*- coding: utf-8 -*-
"""
Neural Networks for classification
"""

import tensorflow as tf
from keras.models import Sequential
import tensorflow as tf
import pandas as pd
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import sklearn.datasets as data


digits = data.load_digits()

X = digits.data
Y = digits.target


#%% Show a sample of data
idx = 7# np.random.randint(len(Y))
plt.imshow(digits.images[idx],cmap=plt.cm.gray_r)
plt.title('Digit: %d'%(Y[idx]))
plt.show()



#%%

# Spliting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
Y_train_dum = pd.get_dummies(Y_train,prefix='y')
Y_test_dum = pd.get_dummies(Y_test,prefix='y')



#%% Build the neural model

# Neural network structure
model = Sequential([tf.keras.Input(shape = (64,),name='input_layer'),
                    Dense(10, activation='sigmoid'),
                    Dense(10, activation='tanh'),
                    #Dense(5, activation='softmax'),
                    Dense(10, activation='sigmoid')
                    ])

#  remove backpropagation
#for layer in model.layers:
#    layer.trainable = False


# Optimizer configuration
#model.compile(loss='binary_crossentropy',
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Neural network training
model_history=model.fit(X_train, Y_train_dum,epochs=20, batch_size=1, verbose=1)

#%% View the training performance
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,10))
plt.subplot(121)
plt.plot(model_history.history['loss'])
plt.xlabel('Epochs'),plt.ylabel('Loss function')
plt.subplot(122)
plt.plot(model_history.history['accuracy'])
plt.xlabel('Epochs'),plt.ylabel('Accuracy function')


#%% Neural network weights
model.layers[0].get_weights()
# model.get_config() #model configuration

#%% View the model
from keras.utils import plot_model
plot_model(model)
# plot_model(model, to_file='../figures/P9_fig/model.png', show_shapes=True)

plot_model(model, show_shapes=True, show_layer_names=True)



# #%% Generate the Graphviz representation of the model
#from graphviz import Digraph
#dot = Digraph()

#for layer in model.layers:
 #   if isinstance(layer, tf.keras.layers.InputLayer):
  #      dot.node(layer.name, label=f"Input Layer\n{layer.input_shape[1:]}")
   # else:
    #    dot.node(layer.name, label=f"{layer.name}\n{layer.output_shape[1:]}")

#for i in range(len(model.layers) - 1):
 #   dot.edge(model.layers[i].name, model.layers[i+1].name)

# # Save the graph as a PDF
# #dot.render("model_graph", format="pdf", cleanup=True)

# # Plot the graph
#dot


#%% Use the model

# Latest version
Y_prob = model.predict(X_test)
Y_pred = np.argmax(Y_prob, axis=1)

#%% View the neural network prediction
indx = 5
fig = plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(np.reshape(X_test[indx,:],(8,8)),cmap=plt.cm.gray_r)
plt.subplot(122)
plt.bar(np.arange(10),Y_prob[indx,:])
plt.ylim([0,1])
plt.title('Prediction: %d'%np.argmax(Y_prob[indx,:]))

#%% Evaluating the model
score = model.evaluate(X_test, Y_test_dum,verbose=1)
print(score)

#%% More metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score)
Yhat_train = np.argmax(model.predict(X_train),axis=1)
Yhat_test = np.argmax(model.predict(X_test),axis=1)


accu_train = accuracy_score(Y_train,Yhat_train)
prec_train = precision_score(Y_train,Yhat_train,average='weighted')
reca_train = recall_score(Y_train,Yhat_train,average='weighted')

accu_test = accuracy_score(Y_test,Yhat_test)
prec_test = precision_score(Y_test,Yhat_test,average='weighted')
reca_test = recall_score(Y_test,Yhat_test,average='weighted')

print(' \t\t Accu \t Prec \t Reca\n Train \t %0.3f \t %0.3f \t %0.3f\n  Test \t %0.3f \t %0.3f \t %0.3f'%(accu_train,prec_train,reca_train,accu_test,prec_test,reca_test))


#%%

