# -*- coding: utf-8 -*-

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
import scipy.optimize as opt


#%% EXAMPLE 1. A simple logistic regression
#%% Creating the dataset 
X,Y = make_blobs(n_samples=200,
                 centers=[[0,0],[3,3]],
                 cluster_std=0.5,
                 n_features=2)

plt.scatter(X[:,0],X[:,1],c=Y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


#%% Logistic function
def fun_log(X):
    return 1/(1+np.exp(-X))

#%% Logistic regression function
def reg_log(W,X,Y):
    V = np.matrix(X)*np.matrix(W).transpose()
    return np.array(fun_log(V))[:,0]

#%% Cost function
def fun_cost(W,X,Y):
    Yhat = reg_log(W,X,Y)
    J = np.sum(-Y*np.log(Yhat)-(1-Y)*np.log(1-Yhat))/len(Y)
    return J


#%% Optimization variables initialitation
Xa = np.append(np.ones((len(Y),1)),X,axis=1)
m,n = np.shape(Xa)
W = np.zeros(n)

#Optimization
res = opt.minimize(fun_cost,W,args=(Xa,Y))
W = res.x

#%% Logistic regression trained
Yhat = np.round(reg_log(W,Xa,Y),0)


#%% Generating the data mesh for the surface
x = np.arange(-3,5,0.01)
y = np.arange(-3,5,0.01)
Xm,Ym = np.meshgrid(x,y)
m,n = np.shape(Xm)
Xmr = Xm.reshape((-1,1))
Ymr = Ym.reshape((-1,1))

Xtmp = np.append(Xmr,Ymr,axis=1)
Xtmp = np.append(np.ones(Xmr.shape),Xtmp,axis=1)
Ytmp = np.round(reg_log(W,Xtmp,Xmr),0)
# Ytmp = reg_log(W,Xtmp,Xmr)
Z = np.reshape(Ytmp,(m,n))

#% View the decision boundary
plt.contour(Xm,Ym,Z)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()



#%% EXAMPLE 2. Polynomial logistic regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import (accuracy_score,precision_score,
                             recall_score)

#%% Import the dataset
data = pd.read_csv('ex2data2.txt',header=None)
X = data.iloc[:,0:2]
Y = data.iloc[:,2]
plt.scatter(X[0],X[1],c=Y)
plt.show()

#%% Preparación de los datos (crear el polinomio)
ngrado = 3
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(X)

#%% Crear y entrenar la regresión logística
modelo = linear_model.LogisticRegression(C=1)
modelo.fit(Xa,Y)

Yhat = modelo.predict(Xa)

#%% Evaluar el modelo
print('Accuracy score = %0.3f'%accuracy_score(Y,Yhat))
#%
print('Precision score = %0.3f'%precision_score(Y,Yhat))
#%
print('Recall score = %0.3f'%recall_score(Y,Yhat))

#%% View the decision boundary
h = 0.01
xmin,xmax,ymin,ymax = X[0].min(),X[0].max(),X[1].min(),X[1].max()
xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))

Xnew = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()])

Xa_new = poly.fit_transform(Xnew)
Z = modelo.predict(Xa_new)
Z = Z.reshape(xx.shape)

plt.contour(xx,yy,Z)
plt.scatter(X[0],X[1],c=Y)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show()

#%% View the overfiting by means the model coefficients
W = modelo.coef_
plt.bar(np.arange(len(W[0])),W[0])
plt.xticks(np.arange(len(W[0])),poly.get_feature_names_out(), rotation='vertical')
plt.show()

#%% Import the model for later
#% Save the model trained
import pickle
pickle.dump(modelo,open('model.sav','wb'))

#% Load the model saved
model = pickle.load(open('model.sav','rb'))

#%% Create a new model without the features with low coef
threshold = 0.5 # this value is a proposed by the analyst

# Select the predictor with high coef
indx = np.abs(W)>threshold
indx = np.arange(len(W[0]))[indx[0]]
Xa_min = Xa[:,indx]

#% New model with the Xa minima
min_model = linear_model.LogisticRegression(C=1)
min_model.fit(Xa_min,Y)

Yhat_min = min_model.predict(Xa_min)

#%% Compare metrics
print('Metrics for the complete model')
print('Accuracy score = %0.3f'%accuracy_score(Y,Yhat))
print('Precision score = %0.3f'%precision_score(Y,Yhat))
print('Recall score = %0.3f'%recall_score(Y,Yhat))

print('\nMetrics for the minima model')
print('Accuracy score = %0.3f'%accuracy_score(Y,Yhat_min))
print('Precision score = %0.3f'%precision_score(Y,Yhat_min))
print('Recall score = %0.3f'%recall_score(Y,Yhat_min))

#%% Compare the parameters for both models
W = modelo.coef_
W_min =  min_model.coef_
plt.subplot(2,1,1)
plt.bar(np.arange(len(W[0])),W[0])
plt.xticks(np.arange(len(W[0])),poly.get_feature_names_out(), rotation='vertical')
plt.subplot(2,1,2)
plt.bar(np.arange(len(W_min[0])),W_min[0])
plt.xticks(np.arange(len(W_min[0])),np.array(poly.get_feature_names_out())[indx], rotation='vertical')
plt.tight_layout()
plt.show()













