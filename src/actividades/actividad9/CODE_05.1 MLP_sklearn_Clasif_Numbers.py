# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 17:47:31 2025

@author: zaratejo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load digits dataset
digits = load_digits()
X = digits.data
Y = digits.target

# Show a sample digit
idx = 7
plt.imshow(digits.images[idx], cmap=plt.cm.gray_r)
plt.title(f'Digit: {Y[idx]}')
plt.show()

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#%% Build and train the MLP model
model = MLPClassifier(hidden_layer_sizes=(10, 10, 10),
                      activation='tanh',  
                      solver='sgd',
                      max_iter=20,
                      batch_size=1,
                      verbose=True)

model.fit(X_train, Y_train)

# Predict
Yhat_train = model.predict(X_train)
Yhat_test = model.predict(X_test)

# Evaluate
accu_train = accuracy_score(Y_train, Yhat_train)
prec_train = precision_score(Y_train, Yhat_train, average='weighted')
reca_train = recall_score(Y_train, Yhat_train, average='weighted')

accu_test = accuracy_score(Y_test, Yhat_test)
prec_test = precision_score(Y_test, Yhat_test, average='weighted')
reca_test = recall_score(Y_test, Yhat_test, average='weighted')

print(' \t\t Accu \t Prec \t Reca\n Train \t %0.3f \t %0.3f \t %0.3f\n  Test \t %0.3f \t %0.3f \t %0.3f' %
      (accu_train, prec_train, reca_train, accu_test, prec_test, reca_test))

#%% Plot prediction for one sample
indx = 5
Y_prob = model.predict_proba(X_test)
fig = plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(np.reshape(X_test[indx, :], (8, 8)), cmap=plt.cm.gray_r)
plt.subplot(122)
plt.bar(np.arange(10), Y_prob[indx, :])
plt.ylim([0, 1])
plt.title(f'Prediction: {np.argmax(Y_prob[indx, :])}')
