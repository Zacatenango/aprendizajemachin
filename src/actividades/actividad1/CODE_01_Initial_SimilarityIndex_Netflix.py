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
import sklearn.metrics as sklearn_metrics # similarity metrics
import scipy.spatial.distance as scipy_spatial_distance # distance metrics

#%% Import data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Test de películas (anónimo)(1-12).xlsx')
# data = pd.read_excel(file_path,encoding='latin_1',index_col=0) #old versions
data = pd.read_excel(file_path,index_col=0)
data.head()

#%% Sel7ect columns
# Seleccionamos cada 3 columnas a partir de la columna 9, ya que el acomodo de la tabla es
# <título de la película con las calificaciones que cada usuario puso> - <puntos> - <comentarios>
# Y de ahí sólo nos importan las calificaciones
def select_columns(x):
  csel = np.arange(9,246,3)
  users1 = list(x.iloc[:,6])
  cnames1 = list(x.columns.values[csel])
  x = x[cnames1]
  x.index = users1
  
  return x

data_seleccionada =  select_columns(data)
data_seleccionada.head()


#%% Average rating of the movies
# Sacamos unas estadísticas básicas: el promedio de calificación de cada película, y el de cada 
# persona (nos interesa saber si alguien es pesimista)
movie_prom = data_seleccionada.mean(axis=0)
user_prom = data_seleccionada.mean(axis=1)

#%% Change the stars to like or dislike
# Convertimos los datos a binario: tomamos como like una calificación de 4 o más
cnames = list(data_seleccionada.columns.values)
fnames = np.array(data_seleccionada.index)
for col in cnames:
    data_seleccionada[col]=np.where(data_seleccionada[col]>3,1,0)
data_seleccionada.head()
datan_binarizado = data_seleccionada.copy()

#%% Calculate similarity indices in users by sklearn
# Usamos para eso los likes de datan_binarizado. Tomamos <datan_binarizado.iloc[0,:]>, que son las
# calificaciones del usuario 0 (Mary11), como el conjunto "verdadero"; y como el conjunto "predicho"
# tomamos <datan_binarizado.iloc[1,:]>, que son las calificaciones de Walky (usuario 1).
# Lo hacemos para determinar la similitud entre ambos, En los siguientes renglones sacaremos 2
# diferentes índices de similitud.
cf_m = sklearn_metrics.confusion_matrix(datan_binarizado.iloc[0,:],datan_binarizado.iloc[1,:])

# Aquí usamos la similitud simple
# Sean:
# - a = registros falsos en A y B
# - b = registros falsos en A y ciertos en B
# - c = registros ciertos en A y falsos en B
# - d = registros ciertos en A y B
# La similitud simple es (a + b) / (a+b+c+d) (aciertos / total)
sim_simple = sklearn_metrics.accuracy_score(data_seleccionada.iloc[0,:],data_seleccionada.iloc[1,:])
#sim_simple_new = (cf_m[0,0]+cf_m[1,1])/np.sum(cf_m)
print('Simple : %0.4f'%sim_simple)

# Índice de Jaccard: d / (b+c+d) - películas que gustan a ambos / total excluyendo las que no gustan
# Esto limita la similitud a únicamente casos positivos; 
sim_jac = sklearn_metrics.jaccard_score(data_seleccionada.iloc[0,:],data_seleccionada.iloc[1,:])
sim_jac = (cf_m[0,0])/(np.sum(cf_m)-cf_m[1,1])
print('Jaccard: %0.4f'%sim_jac)

cf_m_binaria = cf_m

# Tip for those who have a different syntax
# conda update sklearn

#%% Calculation of distances by scipy
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
# Sacamos 2 tipos de distancia: el teorema de Pitágoras (euclidiana), y la distancia Canberra
d1 = scipy_spatial_distance.euclidean(data_seleccionada.iloc[0,:],data_seleccionada.iloc[5,:])
print('Simple : %0.4f'%d1)
# La distancia Canberra es una variante ponderada de la distancia Manhattan. Sean p = (p1, ..., pn)
# y q = (q1, q2, ..., qn) 2 vectores en R^n; su distancia Canberra es:
#    d(p,q) = sum for i = 1 to n of [ abs(p[i] - q[i]) / ( abs(p[i]) + abs(q[i]) ) ]
# Dividir entre la suma de las dimensiones hace que la distancia calculada varíe mucho cuando las
# dimensiones son muy pequeñas y cercanas al 0 (ya que dividir entre algo muy pequeño da algo muy
# grande), lo que permite detectar cambios minúsculos en dimensiones pequeñas.
d2 = scipy_spatial_distance.canberra(data_seleccionada.iloc[0,:],data_seleccionada.iloc[5,:])
print('Canberra: %0.4f'%d2)

#%% Calculate all possible combinations by scipy
D1 = scipy_spatial_distance.pdist(data_seleccionada,'matching')
D1 = scipy_spatial_distance.squareform(D1)

D2 = scipy_spatial_distance.pdist(data_seleccionada,'jaccard')
D2 = scipy_spatial_distance.squareform(D2)

#%% Select a user and determine the other most similar user
user = 1
D_user = D1[user]
D_user_sort = np.sort(D_user)
indx_user = np.argsort(D_user)


#%% Recommendation version 1. The most similar user
User = data_seleccionada.loc[fnames[user]]
User_sim = data_seleccionada.loc[fnames[indx_user[1]]]

indx_recomen = (User_sim ==1)&(User==0)
recomend1 = list(User.index[indx_recomen])
print('\n Movie list recommended:\n')
print(recomend1)



#%% Recommendation version 2. The k most similar users
k = 5
User = data_seleccionada.loc[fnames[user]]
User_sim = np.mean(data_seleccionada.loc[fnames[indx_user[1:k+1]]],axis=0)
User_sim[User_sim<=0.5] = 0
User_sim[User_sim>0.5] = 1

indx_recomen = (User_sim ==1)&(User==0)
recomend2 = list(User.index[indx_recomen])
print('\n Movie list recommended:\n')
print(recomend2)


#%% SIMILARITY WITH MULTISTATE VARIABLES
data_seleccionada =  select_columns(data)
data_seleccionada.head()
data_seleccionada.fillna(0,inplace=True)

#%% Multistate similarity metrics
cf_m = sklearn_metrics.confusion_matrix(data_seleccionada.iloc[0,:],data_seleccionada.iloc[1,:])
sim_simple = sklearn_metrics.accuracy_score(data_seleccionada.iloc[0,:],data_seleccionada.iloc[1,:])
#sim_simple = sklearn_metrics.accuracy_score(datan.iloc[0,:],datan.iloc[1,:],average='weighted') # old versions
print('Simple : %0.4f'%sim_simple)
sim_jac = sklearn_metrics.jaccard_score(data_seleccionada.iloc[0,:],data_seleccionada.iloc[1,:],average='weighted')
print('Jaccard : %0.4f'%sim_jac)


#%% GENERATION OF AUXILIARY VARIABLES

# Example of a single variable
dummy1 = pd.get_dummies(data_seleccionada[cnames[1]])
# dummy1 = pd.get_dummies(datan[cnames[1]],prefix=cnames[1])

#%% Example with users of the entire table
datan_dummy = pd.get_dummies(data_seleccionada[cnames[0]],prefix=cnames[0])
for col in cnames[1:]:
    tmp = pd.get_dummies(data_seleccionada[col],prefix=col)
    datan_dummy = datan_dummy.join(tmp)
del tmp


#%% DISTANCES WITH QUANTITATIVE VARIABLES
data_seleccionada =  select_columns(data)
data_seleccionada.head()
data_seleccionada.fillna(0,inplace=True)

#%% Euclidean Distance
D1 = scipy_spatial_distance.pdist(data_seleccionada,'euclidean')
D1 = scipy_spatial_distance.squareform(D1)

#%% Cosine Distance
D2 = scipy_spatial_distance.pdist(data_seleccionada,'cosine')
D2 = scipy_spatial_distance.squareform(D2)

#%% Correlation Distance
D3 = scipy_spatial_distance.pdist(data_seleccionada,'correlation')
D3 = scipy_spatial_distance.squareform(D3)
# %%
