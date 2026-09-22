# -*- coding: utf-8 -*-

# Import libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
import sklearn.datasets as data
from sklearn.metrics import accuracy_score

#%% Generating the dataset
digits = data.load_digits()

X = digits.data
Y = digits.target


#%% Show a sample of data
idx = 50# np.random.randint(len(Y))
plt.imshow(digits.images[idx],cmap=plt.cm.gray_r)
plt.title('Digit: %d'%(Y[idx]))
plt.show()


#%% Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.3, random_state=42)

#%% Aplying a simple logistic regressor
model1 = LogisticRegression(solver='lbfgs', max_iter=10000)#Goldfarb–Shanno
model1.fit(X_train,Y_train)
Yhat1_train = model1.predict(X_train)
Yhat1_test = model1.predict(X_test)

print('Training accuracy score: %0.2f'%(accuracy_score(Y_train,Yhat1_train)))
print('Testing accuracy score: %0.2f'%(accuracy_score(Y_test,Yhat1_test)))

#%% Aplying the PCA analisys
from sklearn.decomposition import PCA
pca_model = PCA()
pca_model.fit(X_train)
X_train_pca = pca_model.transform(X_train)
X_test_pca = pca_model.transform(X_test)

# View the covariance ratio
plt.bar(np.arange(len(pca_model.explained_variance_ratio_)),pca_model.explained_variance_ratio_)
plt.xlabel('Num eigenvalues')
plt.ylabel('% explained variance')

plt.show()

#%% Selecting the variables with the PCA
threshold = 0.9
idx = np.cumsum(pca_model.explained_variance_ratio_)<=threshold
X_train_pca = X_train_pca[:,idx]
X_test_pca = X_test_pca[:,idx]
plt.bar(np.arange(len(pca_model.explained_variance_ratio_)),np.cumsum(pca_model.explained_variance_ratio_))
plt.hlines(threshold,0, 64,'r')
plt.xlabel('Num eigenvalues')
plt.ylabel('% explained variance')
plt.title('%d components must be selected at least'%(np.sum(idx)))

#%% Aplying the LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


lda_model = LDA(store_covariance=True, solver='svd')
lda_model = lda_model.fit(X_train,Y_train)


#‘svd’: Singular value decomposition (default). Does not compute the covariance matrix, therefore this solver is recommended for data with a large number of features.
#‘lsqr’: Least squares solution. Can be combined with shrinkage or custom covariance estimator.
#‘eigen’: Eigenvalue decomposition. Can be combined with shrinkage or custom covariance estimator.


# View the covariance ratio
plt.bar(np.arange(len(lda_model.explained_variance_ratio_)),lda_model.explained_variance_ratio_)
plt.xlabel('Num eigenvalues')
plt.ylabel('% explained variance')
plt.show()
#%% Data transform with LDA

X_train_lda = lda_model.transform(X_train)
X_test_lda = lda_model.transform(X_test)

Yhat_lda_train = lda_model.predict(X_train)
Yhat_lda_test = lda_model.predict(X_test)

print('Training accuracy score: %0.2f'%(accuracy_score(Y_train,Yhat_lda_train)))
print('Testing accuracy score: %0.2f'%(accuracy_score(Y_test,Yhat_lda_test)))


#%% PCA+ logistic regression
model2 = LogisticRegression(solver='lbfgs', max_iter=10000)#Goldfarb–Shanno
model2.fit(X_train_pca,Y_train)
Yhat2_train_PCA = model2.predict(X_train_pca)
Yhat2_test_PCA = model2.predict(X_test_pca)

print('Training accuracy score: %0.2f'%(accuracy_score(Y_train,Yhat2_train_PCA)))
print('Testing accuracy score: %0.2f'%(accuracy_score(Y_test,Yhat2_test_PCA)))





#%% Show a sample of reduced data by PCA
idx = np.random.randint(len(Y_train))
fig = plt.figure(figsize=(10,12))
for k in range(25):
    plt.subplot(5,5,k+1)
    plt.imshow(np.reshape(X_train_pca[k,:],(4,5)),cmap=plt.cm.gray_r)
    plt.title('Digit: %d'%(Yhat2_train_PCA[k]))
plt.show()





#%% LDA+ logistic regression
model3 = LogisticRegression(solver='lbfgs', max_iter=10000)#Goldfarb–Shanno
model3.fit(X_train_lda,Y_train)
Yhat3_train_LDA_log = model3.predict(X_train_lda)
Yhat3_test_LDA_log = model3.predict(X_test_lda)

print('Training accuracy score: %0.2f'%(accuracy_score(Y_train,Yhat3_train_LDA_log)))
print('Testing accuracy score: %0.2f'%(accuracy_score(Y_test,Yhat3_test_LDA_log)))




#%% Test a simple logistic regression with some reductions
n_models =  np.shape(X_train_lda)[1]
accuracy_history = pd.DataFrame(columns=('Train','Test'))
for k in range(n_models):
    model_tmp = LogisticRegression(solver='lbfgs', max_iter=10000)
    model_tmp.fit(X_train_lda[:,0:(k+1)],Y_train)
    accuracy_history.loc[k] = [accuracy_score(Y_train,model_tmp.predict(X_train_lda[:,0:(k+1)])),accuracy_score(Y_test,model_tmp.predict(X_test_lda[:,0:(k+1)]))]

#%% View the performance woth the reduction
plt.plot(accuracy_history.index, accuracy_history['Train'],color='b',label='Accu. train')
plt.plot(accuracy_history.index, accuracy_history['Test'],color='r',label='Accu. test')
plt.xlabel('Num. Components'),plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()




#%% Show a sample of reduced data by LDA
idx = np.random.randint(len(Y_train))
fig = plt.figure(figsize=(10,12))
for k in range(25):
    plt.subplot(5,5,k+1)
    plt.imshow(np.reshape(X_train_lda[k,:],(3,3)),cmap=plt.cm.gray_r)
    plt.title('Digit: %d'%(Y_train[k]))
plt.show()


# #%% Show the transformation matrix
import seaborn as sns
sns.heatmap(lda_model.coef_)

#T_transform = lda_model.coef_
#xmesh, ymesh = np.meshgrid(np.arange(np.shape(T_transform)[1]), np.arange(np.shape(T_transform)[0]))
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(xmesh, ymesh, T_transform, marker='o')


#%% Save the models to use them in another application
import pickle

# 1. Logistic regression with original data
pickle.dump(model1,open('model1.sav','wb'))

# 2. Logistic regression with PCA transformation
#pickle.dump(pca_model,open('pca_model.sav','wb'))
pickle.dump(model2,open('model2.sav','wb'))

# 3. Logistic regression with LDA transformation
#pickle.dump(lda_model,open('lda_model.sav','wb'))
pickle.dump(model3,open('model3.sav','wb'))

#%% Retrive the model saved


# 1. Logistic regression with original data
model1_ = pickle.load(open('model1.sav','rb'))

# 2. Logistic regression with PCA transformation
#pca_model_ = pickle.load(open('pca_model.sav','rb'))
model2_ = pickle.load(open('model2.sav','rb'))

# 3. Logistic regression with LDA transformation
#lda_model_ = pickle.load(open('lda_model.sav','rb'))
model3_ = pickle.load(open('model3.sav','rb'))



#%%

Yhat1_train_train = model1_.predict(X_train)


# Using the model with new data
Yhat1_test_test = model1.predict(X_test)

# Show some metrics
print('Training accuracy score: %0.2f'%(accuracy_score(Y_train,Yhat1_train_train)))
print('Testing accuracy score: %0.2f'%(accuracy_score(Y_test,Yhat1_test_test)))










