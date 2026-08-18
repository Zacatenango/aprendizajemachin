# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:40:08 2023

@author: zaratejo
"""

# -*- coding: utf-8 -*-
#%% Import libraries
import os
import pandas as pd
import numpy as np
import sklearn.metrics as skm # similarity metrics
import scipy.spatial.distance as sc # distance metrics

#%% Import data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Test de películas (anónimo)(1-12).xlsx')
# data = pd.read_excel(file_path,encoding='latin_1',index_col=0) #old versions
data = pd.read_excel(file_path,index_col=0)
data.head()

#%% Sel7ect columns
def select_columns(x):
  csel = np.arange(9,246,3)
  users1 = list(x.iloc[:,6])
  cnames1 = list(x.columns.values[csel])
  x = x[cnames1]
  x.index = users1
  
  return x

datan =  select_columns(data)
datan.head()


#%% Average rating of the movies
movie_prom = datan.mean(axis=0)
user_prom = datan.mean(axis=1)

#%% Change the stars to like or dislike
cnames = list(datan.columns.values)
fnames = np.array(datan.index)
for col in cnames:
    datan[col]=np.where(datan[col]>3,1,0)
datan.head()
    

#%% Calculate similarity indices in users by sklearn
cf_m = skm.confusion_matrix(datan.iloc[0,:],datan.iloc[1,:])

sim_simple = skm.accuracy_score(datan.iloc[0,:],datan.iloc[1,:])
#sim_simple_new = (cf_m[0,0]+cf_m[1,1])/np.sum(cf_m)
print('Simple : %0.4f'%sim_simple)

sim_jac = skm.jaccard_score(datan.iloc[0,:],datan.iloc[1,:])
sim_jac = (cf_m[0,0])/(np.sum(cf_m)-cf_m[1,1])
print('Jaccard: %0.4f'%sim_jac)

# Tip for those who have a different syntax
# conda update sklearn

#%% Calculation of distances by scipy
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
d1 = sc.euclidean(datan.iloc[0,:],datan.iloc[5,:])
print('Simple : %0.4f'%d1)
d2 = sc.canberra(datan.iloc[0,:],datan.iloc[5,:])
print('canberra: %0.4f'%d2)

#%% Calculate all possible combinations by scipy
D1 = sc.pdist(datan,'matching')
D1 = sc.squareform(D1)

D2 = sc.pdist(datan,'jaccard')
D2 = sc.squareform(D2)

#%% Select a user and determine the other most similar user
user = 1
D_user = D1[user]
D_user_sort = np.sort(D_user)
indx_user = np.argsort(D_user)


#%% Recommendation version 1. The most similar user
User = datan.loc[fnames[user]]
User_sim = datan.loc[fnames[indx_user[1]]]

indx_recomen = (User_sim ==1)&(User==0)
recomend1 = list(User.index[indx_recomen])
print('\n Movie list recommended:\n')
print(recomend1)



#%% Recommendation version 2. The k most similar users
k = 5
User = datan.loc[fnames[user]]
User_sim = np.mean(datan.loc[fnames[indx_user[1:k+1]]],axis=0)
User_sim[User_sim<=0.5] = 0
User_sim[User_sim>0.5] = 1

indx_recomen = (User_sim ==1)&(User==0)
recomend2 = list(User.index[indx_recomen])
print('\n Movie list recommended:\n')
print(recomend2)


#%% SIMILARITY WITH MULTISTATE VARIABLES
datan =  select_columns(data)
datan.head()
datan.fillna(0,inplace=True)

#%% Multistate similarity metrics
cf_m = skm.confusion_matrix(datan.iloc[0,:],datan.iloc[1,:])
sim_simple = skm.accuracy_score(datan.iloc[0,:],datan.iloc[1,:])
#sim_simple = skm.accuracy_score(datan.iloc[0,:],datan.iloc[1,:],average='weighted') # old versions
print('Simple : %0.4f'%sim_simple)
sim_jac = skm.jaccard_score(datan.iloc[0,:],datan.iloc[1,:],average='weighted')
print('Jaccard : %0.4f'%sim_jac)


#%% GENERATION OF AUXILIARY VARIABLES

# Example of a single variable
dummy1 = pd.get_dummies(datan[cnames[1]])
# dummy1 = pd.get_dummies(datan[cnames[1]],prefix=cnames[1])

#%% Example with users of the entire table
datan_dummy = pd.get_dummies(datan[cnames[0]],prefix=cnames[0])
for col in cnames[1:]:
    tmp = pd.get_dummies(datan[col],prefix=col)
    datan_dummy = datan_dummy.join(tmp)
del tmp


#%% DISTANCES WITH QUANTITATIVE VARIABLES
datan =  select_columns(data)
datan.head()
datan.fillna(0,inplace=True)

#%% Euclidean Distance
D1 = sc.pdist(datan,'euclidean')
D1 = sc.squareform(D1)

#%% Cosine Distance
D2 = sc.pdist(datan,'cosine')
D2 = sc.squareform(D2)

#%% Correlation Distance
D3 = sc.pdist(datan,'correlation')
D3 = sc.squareform(D3)
# %%
