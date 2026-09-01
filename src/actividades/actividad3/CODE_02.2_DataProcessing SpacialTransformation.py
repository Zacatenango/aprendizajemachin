# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:29:05 2024

@author: zaratejo
"""


# Import libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% IMPORT THE DATA SET
data = pd.read_csv('../Data/glass/glass.data',header=None)
names = ['ID','Refractive_index','Na','Mg','Al','Si','K',
         'Ca','Ba','Fe','Glass_type']
data.columns = names

#%% SCALING OF VARIABLES BY NORMALIZATION
data['Refractive_index_scale'] = (data.Refractive_index-data.Refractive_index.mean())/data.Refractive_index.std()
data['Na_scale'] = (data.Na-data.Na.mean())/data.Na.std()

### Scaling through scikit-learn
from sklearn import preprocessing
data['Refractive_index_scale'] = preprocessing.scale(data.Refractive_index)
data['Na_scale'] = preprocessing.scale(data.Na)


#%% Function to determine the outliers
def find_boundaries(df_var,distance=1.5):
    IQR = df_var.quantile(0.75)-df_var.quantile(0.25)
    lower = df_var.quantile(0.25)-IQR*distance
    upper = df_var.quantile(0.75)+IQR*distance
    return lower,upper

lmin,lmax = find_boundaries(data['Refractive_index'])
outliers = np.where(data['Refractive_index'] > lmax, True,np.where(data['Refractive_index'] < lmin, True, False))
outliers_df = data.loc[outliers, 'Refractive_index']


#%% Spatial sign transformation to mitigate outliers
tmp = data[['Refractive_index_scale','Na_scale']]
modulo = np.sqrt(np.sum(tmp*tmp,axis=1))
tmp['Refractive_index_scale'] = tmp['Refractive_index_scale']/modulo
tmp['Na_scale'] = tmp['Na_scale']/modulo
plt.scatter(tmp.Refractive_index_scale,tmp.Na_scale)

#%%
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
#The norm to use to normalize each non zero sample
# Using scikit-learn
from sklearn import preprocessing
tmp = preprocessing.normalize(np.array(tmp), norm='l2') #l1,l2, max
fig = plt.figure(figsize=(9,5))
plt.subplot(1,2,1)
plt.scatter(data['Refractive_index_scale'][outliers],data['Na_scale'][outliers],c='r')
plt.scatter(data['Refractive_index_scale'][~outliers],data['Na_scale'][~outliers])
plt.xlabel('Refractive_index_scale'),plt.ylabel('Na_scale')
plt.subplot(1,2,2)
plt.scatter(tmp[outliers,0],tmp[outliers,1],c='r')
plt.scatter(tmp[~outliers,0],tmp[~outliers,1])
plt.xlabel('Refractive_index_scale'),plt.ylabel('Na_scale')
plt.tight_layout()
plt.show()
#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Standardize the features
scaler = StandardScaler()
tmp = data[['Refractive_index_scale','Na_scale']]
scaled_data = tmp# scaler.fit_transform(tmp)

# Apply PCA
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(scaled_data)

fig = plt.figure(figsize=(9,5))
plt.subplot(1,2,1)
plt.scatter(data['Refractive_index_scale'][outliers],data['Na_scale'][outliers],c='r')
plt.scatter(data['Refractive_index_scale'][~outliers],data['Na_scale'][~outliers])
plt.xlabel('Refractive_index_scale'),plt.ylabel('Na_scale')
plt.subplot(1,2,2)
plt.scatter(transformed_data[outliers,0], transformed_data[outliers,1],c='r')
plt.scatter(transformed_data[~outliers,0], transformed_data[~outliers,1])
plt.xlabel('PCA Component 1'),plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.show()

#%%
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
#PCA:linear technique
# t-SNE is nonlinear and focuses on preserving the local structure useful for visualizing datasets in 2 or 3 dimensions
# Standardize the features
scaler = StandardScaler()
tmp = data[['Refractive_index_scale','Na_scale']]
scaled_data = tmp# scaler.fit_transform(tmp)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(scaled_data)

fig = plt.figure(figsize=(9,5))
plt.subplot(1,2,1)
plt.scatter(data['Refractive_index_scale'][outliers],data['Na_scale'][outliers],c='r')
plt.scatter(data['Refractive_index_scale'][~outliers],data['Na_scale'][~outliers])
plt.xlabel('Refractive_index_scale'),plt.ylabel('Na_scale')
plt.subplot(1,2,2)
plt.scatter(transformed_data[outliers,0], transformed_data[outliers,1],c='r')
plt.scatter(transformed_data[~outliers,0], transformed_data[~outliers,1])
plt.xlabel('t-SNE Component 1'),plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.show()