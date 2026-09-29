# -*- coding: utf-8 -*-
"""
@author: riemannruiz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import (accuracy_score,precision_score,
                             recall_score,confusion_matrix,f1_score)

#%% Performance evaluation function
def eval_perform(Y,Yhat):
    accu = accuracy_score(Y,Yhat)
    prec = precision_score(Y,Yhat,average='weighted')
    reca = recall_score(Y,Yhat,average='weighted')
    f1_train = f1_score(Y,Yhat)
    print('\n\t\t Accu \t Prec \t Reca \t F1\nEval \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu, prec, reca, f1_train))



#%% Import data
# data = pd.read_csv('../Data/ex2data1.txt',header=None)
# data = pd.read_csv('../Data/ex2data2.txt',header=None)
# X = data.iloc[:,0:2]
# Y = data.iloc[:,2]
# plt.scatter(X[0],X[1],c=Y)
# plt.show()

#%% New example dataset
# Generate the dataset
np.random.seed(103)
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
X = np.r_[np.random.randn(20,2)-[1,1],np.random.randn(20,2)]
Y = np.array([0]*20 + [1]*20)

# View the dataset
indx = Y==1
fig = plt.figure(figsize=(8,8))
plt.scatter(X[indx,0],X[indx,1],label='Class: 1')
plt.scatter(X[~indx,0],X[~indx,1],label='Class: 0')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
#plt.grid()
#fig.savefig('../figures/fig2_dataset_14.png')
plt.show()

#%% Select a polynomial degree
model = linear_model.LogisticRegression()
ndegree = 2
poly = PolynomialFeatures(ndegree)
Xa = poly.fit_transform(X)
model.fit(Xa,Y)
Yhat = model.predict(Xa)
# Model evaluation
cfm = confusion_matrix(Y,Yhat).T
accu_train = accuracy_score(Y,Yhat)
prec_train = precision_score(Y,Yhat)
reca_train = recall_score(Y,Yhat)
f1_train = f1_score(Y,Yhat)

print('\nTrained Model\n\t\t Accu \t Prec \t Reca \t F1\n Train \t %0.3f \t %0.3f \t %0.3f \t %0.3f\n '%(accu_train,prec_train,reca_train,f1_train))


#%% View the separation surface
h = 0.1
xmin, xmax, ymin, ymax = X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()
xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))

Xnew = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
Xa_new = poly.fit_transform(Xnew)

# Assuming model is already defined and fitted
# Replace `model` with your model instance
Z = model.predict(Xa_new)
Z = Z.reshape(xx.shape)

thres = 0.5
Zd = model.predict_proba(Xa_new)[:,0]
Zd = Zd > thres  # Using the defined threshold
Zd = Zd.reshape(xx.shape)

fig, ax = plt.subplots()
CS = ax.contour(xx, yy, Zd)
ax.clabel(CS, inline=1, fontsize=10)
plt.scatter(Xnew.iloc[:,0], Xnew.iloc[:,1], c=Z.flatten(), cmap='PuBuGn', alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=Y, cmap='Greys')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('x_p1')
plt.ylabel('x_p2')
plt.show()


#%% ROC CURVES
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(np.array(Y), np.array(model.predict_proba(Xa)[:,1]))

points = [5, 8, min(14, len(fpr)-1)]  # the last point 

plt.figure(figsize=(12,8))
plt.plot(fpr, tpr)
for k in points:
    plt.text(fpr[k], tpr[k], '     Threshold: %0.3f (Spec: %0.3f, Sens: %0.3f)' % (thresholds[k], 1-fpr[k], tpr[k]))
plt.scatter(fpr[points], tpr[points])
plt.title('ROC curve')
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.show()


#%% Obtain the best threshold to maximize sensitivity and specificity
dist = np.sqrt(np.power(fpr,2)+np.power(1-tpr,2))
indx = np.argmin(dist)
print('\nBest Threshold:\t %0.3f \nSpecificity:\t %0.3f\nSensitivity:\t %0.3f\n '%(thresholds[indx],1-fpr[indx],tpr[indx]))

plt.figure()
plt.plot(thresholds,1-fpr,label='Specificity')
plt.plot(thresholds,tpr,label='Sensitivity')
plt.vlines(thresholds[indx], 0, 1, label='Best Threshold', color='r')
plt.legend()
plt.show()

#%% Comparing a new model
model2 = linear_model.LogisticRegression()
ndegree2 = 3
poly2 = PolynomialFeatures(ndegree2)
Xa2 = poly2.fit_transform(X)
model2.fit(Xa2,Y)
fpr2, tpr2, thresholds2 = roc_curve(np.array(Y), np.array(model2.predict_proba(Xa2)[:,1]))

# Model evaluation
cfm = confusion_matrix(Y,Yhat).T
accu_train = accuracy_score(Y,Yhat)
prec_train = precision_score(Y,Yhat)
reca_train = recall_score(Y,Yhat)
f1 = f1_score(Y,Yhat)

print('\nTrained Model\n\t\t Accu \t Prec \t Reca \t F1\n Train \t %0.3f \t %0.3f \t %0.3f \t %0.3f\n' % (accu_train, prec_train, reca_train, f1))

#%% View both separation surfaces
Xa_new = poly2.fit_transform(Xnew)
Z2 = model2.predict(Xa_new)
Z2 = Z2.reshape(xx.shape)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.contour(xx, yy, Z, label='RLog degree %d' % ndegree)
plt.scatter(X[:,0], X[:,1], c=Y)  # Corrected scatter plot
plt.title('RLog degree %d' % ndegree)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('x_p1')
plt.ylabel('x_p2')
plt.subplot(1,2,2)
plt.contour(xx, yy, Z2, label='RLog degree %d' % ndegree2)
plt.scatter(X[:,0], X[:,1], c=Y)  # Corrected scatter plot
plt.title('RLog degree %d' % ndegree2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('x_p1')
plt.ylabel('x_p2')
plt.tight_layout()
plt.show()

#%% View the ROC Curves of both models
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label='RLog degree %d'%ndegree)
plt.plot(fpr2,tpr2,label='RLog degree %d'%ndegree2)
plt.legend()
plt.title('ROC curve')
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.show()

#%% Comparing models with AUC
from sklearn.metrics import auc
model2 = linear_model.LogisticRegression()
ndegree2 = 3
poly2 = PolynomialFeatures(ndegree2)
Xa2 = poly2.fit_transform(X)
model2.fit(Xa2,Y)
fpr2, tpr2, thresholds2 = roc_curve(np.array(Y), np.array(model2.predict_proba(Xa2)[:,1]))

plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label='RLog grado %d, (AUC= %0.3f)'%(ndegree,auc(fpr,tpr)))
plt.plot(fpr2,tpr2,label='RLog grado %d, (AUC= %0.3f)'%(ndegree2,auc(fpr2,tpr2)))
plt.legend()
plt.title('ROC curve')
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.show()

#%% Evaluation with the metrics
eval_perform(Y,model.predict(Xa))
eval_perform(Y,model2.predict(Xa2))