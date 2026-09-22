# -*- coding: utf-8 -*-

# Import libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.datasets as data
from sklearn.metrics import accuracy_score, confusion_matrix

#%% Generating the dataset
digits = data.load_digits()

X = digits.data
Y = digits.target


#%% Show a sample of data
idx = 90# np.random.randint(len(Y))
plt.imshow(digits.images[idx],cmap=plt.cm.gray_r)
plt.title('Digit: %d'%(Y[idx]))
plt.show()


#%% Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.3, random_state=42)

#%% Baseline: logistic regression with original data (for comparison)
baseline = LogisticRegression(solver='lbfgs', max_iter=10000)
baseline.fit(X_train,Y_train)

print('--- Logistic regression (baseline, raw pixels) ---')
print('Training accuracy score: %0.4f'%(accuracy_score(Y_train,baseline.predict(X_train))))
print('Testing accuracy score: %0.4f'%(accuracy_score(Y_test,baseline.predict(X_test))))

#%% 1. SVC with the original data
# The RBF kernel is distance based, so the 0-16 pixel values are standardized
# first. The scaler lives inside the Pipeline, so the pickled object is able to
# classify raw 64-pixel images on its own.
model1 = Pipeline([('scaler', StandardScaler()),
                   ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))])
model1.fit(X_train,Y_train)
Yhat1_train = model1.predict(X_train)
Yhat1_test = model1.predict(X_test)

print('--- SVC with original data ---')
print('Training accuracy score: %0.4f'%(accuracy_score(Y_train,Yhat1_train)))
print('Testing accuracy score: %0.4f'%(accuracy_score(Y_test,Yhat1_test)))

#%% Aplying the PCA analisys
from sklearn.decomposition import PCA
pca_model = PCA()
pca_model.fit(X_train)

# View the covariance ratio
plt.bar(np.arange(len(pca_model.explained_variance_ratio_)),pca_model.explained_variance_ratio_)
plt.xlabel('Num eigenvalues')
plt.ylabel('% explained variance')
plt.show()

#%% Selecting the number of components with the PCA
threshold = 0.9
n_pca = int(np.sum(np.cumsum(pca_model.explained_variance_ratio_)<=threshold))

plt.bar(np.arange(len(pca_model.explained_variance_ratio_)),np.cumsum(pca_model.explained_variance_ratio_))
plt.hlines(threshold,0, 64,'r')
plt.xlabel('Num eigenvalues')
plt.ylabel('% explained variance')
plt.title('%d components must be selected at least'%(n_pca))
plt.show()

#%% 2. PCA + SVC
# PCA(n_components=n_pca) inside the pipeline replaces the boolean mask used in
# Code_03: same 20 components, but refitted and stored together with the SVC.
model2 = Pipeline([('pca', PCA(n_components=n_pca)),
                   ('scaler', StandardScaler()),
                   ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))])
model2.fit(X_train,Y_train)
Yhat2_train_PCA = model2.predict(X_train)
Yhat2_test_PCA = model2.predict(X_test)

print('--- SVC with PCA transformation (%d components) ---'%(n_pca))
print('Training accuracy score: %0.4f'%(accuracy_score(Y_train,Yhat2_train_PCA)))
print('Testing accuracy score: %0.4f'%(accuracy_score(Y_test,Yhat2_test_PCA)))

#%% Show a sample of reduced data by PCA
X_train_pca = model2.named_steps['pca'].transform(X_train)
fig = plt.figure(figsize=(10,12))
for k in range(25):
    plt.subplot(5,5,k+1)
    plt.imshow(np.reshape(X_train_pca[k,:],(4,5)),cmap=plt.cm.gray_r)
    plt.title('Digit: %d'%(Yhat2_train_PCA[k]))
plt.show()

#%% Aplying the LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_model = LDA(store_covariance=True, solver='svd')
lda_model = lda_model.fit(X_train,Y_train)

# LDA is capped at n_classes-1 = 9 components
n_lda = lda_model.transform(X_train).shape[1]

plt.bar(np.arange(len(lda_model.explained_variance_ratio_)),lda_model.explained_variance_ratio_)
plt.xlabel('Num eigenvalues')
plt.ylabel('% explained variance')
plt.show()

#%% 3. LDA + SVC
model3 = Pipeline([('lda', LDA(solver='svd')),
                   ('scaler', StandardScaler()),
                   ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))])
model3.fit(X_train,Y_train)
Yhat3_train_LDA = model3.predict(X_train)
Yhat3_test_LDA = model3.predict(X_test)

print('--- SVC with LDA transformation (%d components) ---'%(n_lda))
print('Training accuracy score: %0.4f'%(accuracy_score(Y_train,Yhat3_train_LDA)))
print('Testing accuracy score: %0.4f'%(accuracy_score(Y_test,Yhat3_test_LDA)))

#%% Show a sample of reduced data by LDA
X_train_lda = model3.named_steps['lda'].transform(X_train)
fig = plt.figure(figsize=(10,12))
for k in range(25):
    plt.subplot(5,5,k+1)
    plt.imshow(np.reshape(X_train_lda[k,:],(3,3)),cmap=plt.cm.gray_r)
    plt.title('Digit: %d'%(Y_train[k]))
plt.show()

#%% Test the SVC with an increasing number of LDA components
accuracy_history = pd.DataFrame(columns=('Train','Test'))
for k in range(n_lda):
    model_tmp = Pipeline([('scaler', StandardScaler()),
                          ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))])
    model_tmp.fit(X_train_lda[:,0:(k+1)],Y_train)
    accuracy_history.loc[k] = [accuracy_score(Y_train,model_tmp.predict(X_train_lda[:,0:(k+1)])),
                               accuracy_score(Y_test,model_tmp.predict(model3.named_steps['lda'].transform(X_test)[:,0:(k+1)]))]

#%% View the performance with the reduction
plt.plot(accuracy_history.index, accuracy_history['Train'],color='b',label='Accu. train')
plt.plot(accuracy_history.index, accuracy_history['Test'],color='r',label='Accu. test')
plt.xlabel('Num. Components'),plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

#%% Compare the kernels on the original data
kernel_history = pd.DataFrame(columns=('Train','Test'))
for kernel in ('linear','poly','rbf','sigmoid'):
    model_tmp = Pipeline([('scaler', StandardScaler()),
                          ('svc', SVC(kernel=kernel, gamma='scale'))])
    model_tmp.fit(X_train,Y_train)
    kernel_history.loc[kernel] = [accuracy_score(Y_train,model_tmp.predict(X_train)),
                                  accuracy_score(Y_test,model_tmp.predict(X_test))]

print('--- SVC kernel comparison (original data) ---')
print(kernel_history)

#%% Confusion matrix of the SVC with original data
import seaborn as sns
sns.heatmap(confusion_matrix(Y_test,Yhat1_test), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'),plt.ylabel('True')
plt.title('SVC with original data - test set')
plt.show()

#%% Save the models to use them in another application
import pickle

# 1. SVC with original data
pickle.dump(model1,open('model_svc1.sav','wb'))

# 2. SVC with PCA transformation
pickle.dump(model2,open('model_svc2.sav','wb'))

# 3. SVC with LDA transformation
pickle.dump(model3,open('model_svc3.sav','wb'))

#%% Retrive the models saved

model1_ = pickle.load(open('model_svc1.sav','rb'))
model2_ = pickle.load(open('model_svc2.sav','rb'))
model3_ = pickle.load(open('model_svc3.sav','rb'))

#%% Using the reloaded models with the data
# Each pipeline carries its own transformer, so the raw X_test can be fed
# directly to any of the three reloaded objects.
print('--- Reloaded models ---')
for name, model in (('model_svc1',model1_),('model_svc2',model2_),('model_svc3',model3_)):
    print('%s -> train: %0.4f  test: %0.4f'%(name,
          accuracy_score(Y_train,model.predict(X_train)),
          accuracy_score(Y_test,model.predict(X_test))))
