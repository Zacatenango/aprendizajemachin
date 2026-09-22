# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,svm
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score)
import pandas as pd

#%% Performance evaluation function
def eval_perform(Y,Yhat):
    accu = accuracy_score(Y,Yhat)
    prec = precision_score(Y,Yhat,average='weighted')
    reca = recall_score(Y,Yhat,average='weighted')
    print('\n \t Accu \t Prec \t Reca\n Eval \t %0.3f \t %0.3f \t %0.3f'%(accu,prec,reca))


#%% Generate the dataset (EXAMPLE 1)
np.random.seed(103)
#X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)]
Y = np.array([0]*20 + [1]*20)

#%% View the dataset
indx = Y==1
fig = plt.figure(figsize=(8,8))
plt.scatter(X[indx,0],X[indx,1],c='g',label='Class: 1')
plt.scatter(X[~indx,0],X[~indx,1],c='r',label='Class: -1')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.grid()
plt.show()

#%% Creating the SV model
modelo = svm.SVC(kernel='linear',C=1)
modelo.fit(X,Y)

Yhat = modelo.predict(X)

eval_perform(Y,Yhat)

#%% View the decision boundary
w = modelo.coef_[0] #Obtain the weights W of the hyperplane
m = -w[0]/w[1]
xx = np.linspace(-5,5)
yy = m*xx-(modelo.intercept_[0]/w[1])

vs = modelo.support_vectors_ #Obtaining the support vectors

# Obtain the the upper and lower support vector
w_norm = w/np.linalg.norm(w)
gamma_sv = np.dot(vs,w_norm)
idx_min = np.argmin(gamma_sv)
idx_max = np.argmax(gamma_sv)

# Create the upper and lower parallel hyperplane
b = vs[idx_min]
yy_down = m*xx + (b[1]-m*b[0])

b = vs[idx_max]
yy_up = m*xx + (b[1]-m*b[0])



indx = Y==1
fig = plt.figure(figsize=(8,8))
plt.scatter(X[indx,0],X[indx,1],c='g',label='Class: 1')
plt.scatter(X[~indx,0],X[~indx,1],c='r',label='Class: -1')
plt.plot(xx,yy,'k-')
plt.scatter(vs[:,0],vs[:,1],s=60,marker='x',facecolors='k')
plt.plot(xx,yy_down,'k--')
plt.plot(xx,yy_up,'k--')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.axis([-5,5,-5,5])
plt.legend()
plt.grid()
plt.show()


#%% Import data (EXAMPLE 2)
data = pd.read_csv('../Data/ex2data2.txt',header=None)
X = data.iloc[:,0:2]
Y = data.iloc[:,2]

#%% Data visualization
fig = plt.figure(figsize=(8,8))
indx = Y==1
plt.scatter(X[0][indx],X[1][indx],c='g',label='Class: +1')
plt.scatter(X[0][~indx],X[1][~indx],c='r',label='Class: -1')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
# fig.savefig('../figures/fig1_svm_2d.png')
plt.show()


#%% Creating the SV model
#C
#Regularization parameter. The strength of the regularization is inversely proportional to C.
# Must be strictly positive. The penalty is a squared l2 penalty. The squared L2 norm
#the regularization term make the SVM less susceptible to outliers to improve generalization.

#modelo = svm.SVC(kernel='linear',C=1, probability=True)
#modelo = svm.SVC(kernel='poly',degree=3,C=1,  probability=True)
modelo = svm.SVC(kernel='rbf',C=1,gamma='auto', probability=True)
#if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
#if ‘auto’, uses 1 / n_features.

# It is posible to obtain a probability metric
#modelo = svm.SVC(kernel='rbf',C=1,gamma='auto',probability=True)

modelo.fit(X,Y)

Yhat = modelo.predict(X)

eval_perform(Y,Yhat)

Yhat_prob = modelo.predict_proba(X)

#%% View the decision boundary
h = 0.1
xmin,xmax,ymin,ymax = X[0].min(),X[0].max(),X[1].min(),X[1].max()
xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))

Xnew = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()])

Z = modelo.predict(Xnew)
Z = Z.reshape(xx.shape)

vs = modelo.support_vectors_

indx = Y==1
fig = plt.figure(figsize=(8,8))
plt.scatter(X[0][indx],X[1][indx],c='g',label='Class: +1')
plt.scatter(X[0][~indx],X[1][~indx],c='r',label='Class: -1')
plt.contour(xx,yy,Z)
plt.scatter(vs[:,0],vs[:,1],s=60,marker='x',facecolors='k')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
# fig.savefig('../figures/fig2_svm_2d.png')
plt.show()
