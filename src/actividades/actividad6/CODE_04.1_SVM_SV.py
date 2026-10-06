# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,svm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
    return accu,prec,reca


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
data = pd.read_csv('ex2data2.txt',header=None)
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


#%% Split into train and test sets
# The test set is kept out of the grid search to get an unbiased estimate
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,
                                                 stratify=Y,random_state=0)

#%% Grid search over the SV model hyperparameters
#C
#Regularization parameter. The strength of the regularization is inversely proportional to C.
# Must be strictly positive. The penalty is a squared l2 penalty. The squared L2 norm
#the regularization term make the SVM less susceptible to outliers to improve generalization.
#gamma
#if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
#if 'auto', uses 1 / n_features.

# One grid per kernel, so each kernel only gets the parameters it uses
param_grid = [
    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]},
    {'kernel': ['poly'], 'C': [0.01, 0.1, 1, 10, 100], 'degree': [2, 3, 4],
     'gamma': ['scale', 'auto']},
    {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100],
     'gamma': ['scale', 'auto', 0.1, 1, 10]},
]

# Same folds for every model, so their CV scores are comparable
cv = StratifiedKFold(n_splits=5)

# It is posible to obtain a probability metric with probability=True
grid_search = GridSearchCV(estimator=svm.SVC(probability=True,random_state=0),
                           param_grid=param_grid,
                           scoring='accuracy', cv=cv,
                           refit=True, return_train_score=True, n_jobs=-1)
grid_search.fit(X_train,Y_train)

#%% Grid search results (top 10 combinations)
results = pd.DataFrame(grid_search.cv_results_)
cols = ['param_kernel','param_C','param_gamma','param_degree',
        'mean_train_score','mean_test_score','std_test_score','rank_test_score']
print(results[cols].sort_values('rank_test_score').head(10).to_string(index=False))

#%% View the CV accuracy for the rbf kernel (C vs gamma)
rbf = results[results['param_kernel']=='rbf']
heat = rbf.pivot_table(index='param_gamma',columns='param_C',
                       values='mean_test_score',aggfunc='first')
fig = plt.figure(figsize=(8,6))
plt.imshow(heat.values,cmap='viridis',aspect='auto')
plt.colorbar(label='Mean CV accuracy')
plt.xticks(range(heat.shape[1]),heat.columns)
plt.yticks(range(heat.shape[0]),heat.index)
plt.xlabel('C')
plt.ylabel('gamma')
plt.title('Grid search: rbf kernel')
plt.show()

#%% Select the model with the best CV score
# Best combination of each kernel, then the kernel with the highest mean CV accuracy
best_per_kernel = (results.sort_values('rank_test_score')
                          .groupby('param_kernel').head(1)
                          .set_index('param_kernel'))
print(best_per_kernel[['params','mean_test_score','std_test_score']].to_string())

best_kernel = best_per_kernel['mean_test_score'].idxmax()
best_params = best_per_kernel.loc[best_kernel,'params']
print('\nSelected model: SVC',best_params)
print('CV accuracy: %0.3f'%best_per_kernel.loc[best_kernel,'mean_test_score'])

# Refit the selected model on the whole training set
modelo = svm.SVC(probability=True,random_state=0,**best_params)
modelo.fit(X_train,Y_train)

Yhat_prob = modelo.predict_proba(X)

#%% Linear regression model
# Linear regression predicts a continuous value, so it is used as a classifier
# by thresholding its output at 0.5
def linreg_classify(model,X):
    return (model.predict(X)>=0.5).astype(int)

# CV accuracy on the same folds used by the grid search
cv_scores_lr = []
for idx_tr,idx_va in cv.split(X_train,Y_train):
    lr = linear_model.LinearRegression()
    lr.fit(X_train.iloc[idx_tr],Y_train.iloc[idx_tr])
    cv_scores_lr.append(accuracy_score(Y_train.iloc[idx_va],
                                       linreg_classify(lr,X_train.iloc[idx_va])))

modelo_lr = linear_model.LinearRegression()
modelo_lr.fit(X_train,Y_train)

#%% Compare the selected SV model with linear regression
comparison = []
for name,predict,cv_mean,cv_std in [
        ('SVC '+best_kernel, modelo.predict,
         best_per_kernel.loc[best_kernel,'mean_test_score'],
         best_per_kernel.loc[best_kernel,'std_test_score']),
        ('Linear Regression', lambda X: linreg_classify(modelo_lr,X),
         np.mean(cv_scores_lr), np.std(cv_scores_lr))]:
    print('\n%s - Train set:'%name)
    accu_tr,_,_ = eval_perform(Y_train,predict(X_train))
    print('\n%s - Test set:'%name)
    accu_te,prec_te,reca_te = eval_perform(Y_test,predict(X_test))
    comparison.append({'Model': name, 'Accu_Train': accu_tr,
                       'Accu_CV': cv_mean, 'Std_CV': cv_std,
                       'Accu_Test': accu_te, 'Prec_Test': prec_te,
                       'Reca_Test': reca_te})

comparison = pd.DataFrame(comparison)
print('\n',comparison.round(3).to_string(index=False))

#%% View the decision boundary
h = 0.1
xmin,xmax,ymin,ymax = X[0].min(),X[0].max(),X[1].min(),X[1].max()
xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))

Xnew = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()])

Z = modelo.predict(Xnew)
Z = Z.reshape(xx.shape)

Z_lr = linreg_classify(modelo_lr,Xnew).reshape(xx.shape)

vs = modelo.support_vectors_

indx = Y==1
fig = plt.figure(figsize=(8,8))
plt.scatter(X[0][indx],X[1][indx],c='g',label='Class: +1')
plt.scatter(X[0][~indx],X[1][~indx],c='r',label='Class: -1')
plt.contour(xx,yy,Z,levels=[0.5],colors='k')
plt.contour(xx,yy,Z_lr,levels=[0.5],colors='b',linestyles='--')
plt.scatter(vs[:,0],vs[:,1],s=60,marker='x',facecolors='k',label='Support vectors')
plt.plot([],[],'k-',label='SVC '+best_kernel)
plt.plot([],[],'b--',label='Linear Regression')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
# fig.savefig('../figures/fig2_svm_2d.png')
plt.show()
