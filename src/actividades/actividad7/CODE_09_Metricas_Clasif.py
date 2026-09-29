# -*- coding: utf-8 -*-

#General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Neural network
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

#Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Preprocessing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score)
from sklearn.metrics import (roc_curve,auc)

#%% Load the data
data = pd.read_csv('../Data/diabetes/diabetes.csv', delimiter=',')

"""

All patients here are females at least 21 years old of Pima Indian heritage.

PregnanciesNumber of times pregnant
GlucosePlasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressureDiastolic blood pressure (mm Hg)
SkinThicknessTriceps skin fold thickness (mm)
Insulin2-Hour serum insulin (mu U/ml)
BMIBody mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunctionDiabetes pedigree function
AgeAge (years)
OutcomeClass variable (0 or 1) 268 of 768 are 1, the others are 0
"""

#%% Description of the data
desc = data.describe()
info = data.info()

#%% Select the data for training and test
X = data.iloc[:,0:8]
Y = np.ravel(data['Outcome'])

# Data scaling
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#%% Neural network construction

# Neural network structure
m_nn = Sequential()
m_nn.add(Dense(8, activation='tanh', input_shape=(8,)))
m_nn.add(Dense(1, activation='sigmoid'))

# Configure the optimizer
m_nn.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Neural network training
m_nn.fit(X_train, Y_train,epochs=20, batch_size=1, verbose=0)

#%% Decision tree construction
m_dt = DecisionTreeClassifier(criterion='gini',
                                splitter='best',
                                max_depth=None)
m_dt = m_dt.fit(X_train, Y_train)

#%% Random Forest construction
m_rf = RandomForestClassifier(n_estimators=50,
                               criterion='gini',
                               max_depth=None,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               #max_features=2,
                               bootstrap=True,
                               oob_score=False,
                               random_state=0,
                               verbose=1)
m_rf = m_rf.fit(X_train, Y_train)

#%% Using the models
# Neural network
Yhat_train_nn = np.round(m_nn.predict(X_train))
Yhat_test_nn = np.round(m_nn.predict(X_test))

# Decision tree
Yhat_train_dt = m_dt.predict(X_train)
Yhat_test_dt = m_dt.predict(X_test)

# Random Forest
Yhat_train_rf = m_rf.predict(X_train)
Yhat_test_rf = m_rf.predict(X_test)


#%% Neural network evaluation
# Train
accu_train_nn = accuracy_score(Y_train,Yhat_train_nn)
prec_train_nn = precision_score(Y_train,Yhat_train_nn)
reca_train_nn = recall_score(Y_train,Yhat_train_nn)
f1_train_nn = f1_score(Y_train,Yhat_train_nn)

#Test
accu_test_nn = accuracy_score(Y_test,Yhat_test_nn)
prec_test_nn = precision_score(Y_test,Yhat_test_nn)
reca_test_nn = recall_score(Y_test,Yhat_test_nn)
f1_test_nn = f1_score(Y_test,Yhat_test_nn)


#%% Decision tree evaluation
# Train
accu_train_dt = accuracy_score(Y_train,Yhat_train_dt)
prec_train_dt = precision_score(Y_train,Yhat_train_dt)
reca_train_dt = recall_score(Y_train,Yhat_train_dt)
f1_train_dt = f1_score(Y_train, Yhat_train_dt)


# Test
accu_test_dt = accuracy_score(Y_test,Yhat_test_dt)
prec_test_dt = precision_score(Y_test,Yhat_test_dt)
reca_test_dt = recall_score(Y_test,Yhat_test_dt)
f1_test_dt = f1_score(Y_test, Yhat_test_dt)


#%% Random Forest evaluation
# Train
accu_train_rf = accuracy_score(Y_train,Yhat_train_rf)
prec_train_rf = precision_score(Y_train,Yhat_train_rf)
reca_train_rf = recall_score(Y_train,Yhat_train_rf)
f1_train_rf = f1_score(Y_train, Yhat_train_rf)


# Test
accu_test_rf = accuracy_score(Y_test,Yhat_test_rf)
prec_test_rf = precision_score(Y_test,Yhat_test_rf)
reca_test_rf = recall_score(Y_test,Yhat_test_rf)
f1_test_rf = f1_score(Y_test, Yhat_test_rf)

#%% View the metrics for the models

print('\nNeural Network')
print('\t\t Accu \t Prec \t Reca \t F1')
print('Train \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu_train_nn, prec_train_nn, reca_train_nn, f1_train_nn))
print('Test \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu_test_nn, prec_test_nn, reca_test_nn, f1_test_nn))

print('\nDecision Tree')
print('\t\t Accu \t Prec \t Reca \t F1')
print('Train \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu_train_dt, prec_train_dt, reca_train_dt, f1_train_dt))
print('Test \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu_test_dt, prec_test_dt, reca_test_dt, f1_test_dt))

print('\nRandom Forest')
print('\t\t Accu \t Prec \t Reca \t F1')
print('Train \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu_train_rf, prec_train_rf, reca_train_rf, f1_train_rf))
print('Test \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu_test_rf, prec_test_rf, reca_test_rf, f1_test_rf))

#%% ROC curves and AUC

# Prediction probabilities
# Neural network
Yprob_train_nn = m_nn.predict(X_train)
Yprob_test_nn = m_nn.predict(X_test)
# Decision tree
Yprob_train_dt = m_dt.predict_proba(X_train)[:,1]
Yprob_test_dt = m_dt.predict_proba(X_test)[:,1]

# Decision tree
Yprob_train_rf = m_rf.predict_proba(X_train)[:,1]
Yprob_test_rf = m_rf.predict_proba(X_test)[:,1]
#%%
#ROC curve and AUC
# Neural Network
fpr_train_nn, tpr_train_nn, thresholds_train_nn = roc_curve(Y_train, Yprob_train_nn)
fpr_test_nn, tpr_test_nn, thresholds_test_nn = roc_curve(Y_test, Yprob_test_nn)
auc_train_nn,auc_test_nn = auc(fpr_train_nn, tpr_train_nn),auc(fpr_test_nn, tpr_test_nn)

# Decision tree
fpr_train_dt, tpr_train_dt, thresholds_train_dt = roc_curve(Y_train, Yprob_train_dt)
fpr_test_dt, tpr_test_dt, thresholds_test_dt = roc_curve(Y_test, Yprob_test_dt)
auc_train_dt,auc_test_dt = auc(fpr_train_dt, tpr_train_dt),auc(fpr_test_dt, tpr_test_dt)

# Random Forest
fpr_train_rf, tpr_train_rf, thresholds_train_rf = roc_curve(Y_train, Yprob_train_rf)
fpr_test_rf, tpr_test_rf, thresholds_test_rf = roc_curve(Y_test, Yprob_test_rf)
auc_train_rf,auc_test_rf = auc(fpr_train_rf, tpr_train_rf),auc(fpr_test_rf, tpr_test_rf)
#%% Show the model comparation
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(fpr_train_nn, tpr_train_nn,label='Neural Network, (AUC= %0.3f)'%auc_train_nn)
plt.plot(fpr_train_dt, tpr_train_dt,label='Decision Tree, (AUC= %0.3f)'%auc_train_dt)
plt.plot(fpr_train_rf, tpr_train_rf,label='Random Forest, (AUC= %0.3f)'%auc_train_rf)
plt.legend()
plt.title('ROC curve train')
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.subplot(1,2,2)
plt.plot(fpr_test_nn, tpr_test_nn,label='Neural Network, (AUC= %0.3f)'%auc_test_nn)
plt.plot(fpr_test_dt, tpr_test_dt,label='Decision Tree, (AUC= %0.3f)'%auc_test_dt)
plt.plot(fpr_test_rf, tpr_test_rf,label='Random Forest, (AUC= %0.3f)'%auc_test_rf)
plt.legend()
plt.title('ROC curve test')
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.savefig('roc_auc.pdf')
plt.show()

#%% Obtain the best threshold to maximize sensitivity and specificity
dist = np.sqrt(np.power(fpr_test_nn,2)+np.power(1-tpr_test_nn,2))
indx = np.argmin(dist)
print('\nBest Threshold:\t %0.3f \nSpecificity:\t %0.3f\nSensitivity:\t %0.3f\n '%(thresholds_test_nn[indx],1-fpr_test_nn[indx],tpr_train_nn[indx]))

plt.figure()
plt.plot(thresholds_test_nn,1-fpr_test_nn,label='Specificity')
plt.plot(thresholds_test_nn,tpr_test_nn,label='Sensitivity')
plt.vlines(thresholds_test_nn[indx], 0, 1, label='Best Threshold', color='r')
plt.legend()
plt.show()


#%% Neural network evaluation
print("Original")
# Train
accu_train_nn = accuracy_score(Y_train,Yhat_train_nn)
prec_train_nn = precision_score(Y_train,Yhat_train_nn)
reca_train_nn = recall_score(Y_train,Yhat_train_nn)
f1_train_nn = f1_score(Y_train, Yhat_train_nn)

#Test
accu_test_nn = accuracy_score(Y_test,Yhat_test_nn)
prec_test_nn = precision_score(Y_test,Yhat_test_nn)
reca_test_nn = recall_score(Y_test,Yhat_test_nn)
f1_test_nn = f1_score(Y_test, Yhat_test_nn)


print('Neural Network\n\t\t Accu \t Prec \t Reca \t F1\n Train \t %0.3f \t %0.3f \t %0.3f \t %0.3f\n  Test \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu_train_nn, prec_train_nn, reca_train_nn, f1_train_nn, accu_test_nn, prec_test_nn, reca_test_nn, f1_test_nn))

print("\nNew")

Yhat_train_nn_adj= np.where(Yprob_train_nn>= thresholds_test_nn[indx],1,0)
Yhat_test_nn_adj= np.where(Yprob_test_nn>= thresholds_test_nn[indx],1,0)

# Train
accu_train_nn_adj = accuracy_score(Y_train,Yhat_train_nn_adj)
prec_train_nn_adj = precision_score(Y_train,Yhat_train_nn_adj)
reca_train_nn_adj = recall_score(Y_train,Yhat_train_nn_adj)
f1_train_nn_adj = f1_score(Y_train, Yhat_train_nn_adj)


#Test
accu_test_nn_adj = accuracy_score(Y_test,Yhat_test_nn_adj)
prec_test_nn_adj = precision_score(Y_test,Yhat_test_nn_adj)
reca_test_nn_adj = recall_score(Y_test,Yhat_test_nn_adj)
f1_test_nn_adj = f1_score(Y_test, Yhat_test_nn_adj)



print('Neural Network\n\t\t Accu \t Prec \t Reca \t F1\n Train \t %0.3f \t %0.3f \t %0.3f \t %0.3f\n  Test \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu_train_nn_adj, prec_train_nn_adj, reca_train_nn_adj, f1_train_nn_adj, accu_test_nn_adj, prec_test_nn_adj, reca_test_nn_adj, f1_test_nn_adj))
