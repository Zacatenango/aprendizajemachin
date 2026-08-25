# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:16:39 2023

@author: zaratejo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:05:32 2019

@author: Rocío Carrasco
"""

import sklearn
import mglearn
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import seaborn as sns

#%% PCA demonsration.  The algorith reduces from two components to 1 at a cost of losing data within the variance
mglearn.plots.plot_pca_illustration()
plt.show()

#%% Load cance DB
cancer=load_breast_cancer()
print(cancer.feature_names)
print(cancer.feature_names.shape)


n=3


#%% Principal Component Analysis from sklearn
pca=PCA(n_components=n)
pca.fit(cancer.data)
transformada=pca.transform(cancer.data)
print(cancer.data.shape)
print(transformada.shape)
#%% Plot some PCA
mglearn.discrete_scatter(transformada[:,0],transformada[:,1], cancer.target)
plt.legend(cancer.target_names,loc='best')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

#%%scalin
escala=MinMaxScaler()
escala.fit(cancer.data)
escalada=escala.transform(cancer.data)
pca.fit(escalada)
transformada=pca.transform(escalada)
mglearn.discrete_scatter(transformada[:,0],transformada[:,1], cancer.target)
plt.legend(cancer.target_names,loc='best')
plt.gca()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()




#%% Validate the "recommended" PCA's
import numpy as np
w,v = np.linalg.eig(pca.get_covariance()) # Get vectors for covariance matrix
w=np.abs(w)
indx = np.argsort(w)[::-1] # sort
porcentaje = w[indx]/np.sum(w) # convert to %
porcent_acum = np.cumsum(porcentaje) # Accumulate

#%% Check Individual PCA in the matrix
matrix_transform = pca.components_.T
heat= pd.DataFrame(pca.components_, columns=cancer.feature_names.tolist())
sns.heatmap(heat,cmap='plasma')

#The transformation matrix has as its columns the eigenvector associated with each principal component. 
#The magnitude of each element of that eigenvector is the contribution of each variable to that principal component.
#Graphically, this contribution can be observed if we plot the eigenvector.
plt.bar(np.arange(30),matrix_transform[:,0])
plt.xlabel('Feature')
plt.ylabel('Magnitude')
plt.show()


