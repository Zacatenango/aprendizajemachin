# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:39:08 2023

@author: zaratejo
"""

# -*- coding: utf-8 -*-

# Import libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% IMPORT THE DATA SET
# A partir de los datos, vamos a intentar predecir el índice de refracción
data = pd.read_csv('glass.data',header=None)
names = ['ID','Refractive_index','Na','Mg','Al','Si','K',
         'Ca','Ba','Fe','Glass_type']
data.columns = names

#%% DATA QUALITY REPORT
def dqr(data):
    
    cols = pd.DataFrame(list(data.columns.values),
                           columns=['Name'],
                           index=list(data.columns.values))
    dtyp = pd.DataFrame(data.dtypes,columns=['Type'])
    misval = pd.DataFrame(data.isnull().sum(),
                                  columns=['N/A value'])
    presval = pd.DataFrame(data.count(),
                                  columns=['Count values'])
    unival = pd.DataFrame(columns=['Unique values'])
    minval = pd.DataFrame(columns=['Min'])
    maxval = pd.DataFrame(columns=['Max'])
    mean =pd.DataFrame(data.mean(), columns=['Mean']) 
    Std =pd.DataFrame(data.std(), columns=['Std']) 
    Var =pd.DataFrame(data.var(), columns=['Var']) 
    median =pd.DataFrame(data.median(), columns=['Median']) 
    
    skewness = pd.DataFrame(data.skew(), columns=['Skewness']) 
    kurtosis = pd.DataFrame(data.kurtosis(), columns=['Kurtosis']) 

    for col in list(data.columns.values):
        unival.loc[col] = [data[col].nunique()]
        try:
            minval.loc[col] = [data[col].min()]
            maxval.loc[col] = [data[col].max()]
        except:
            pass
    
    # Juntar todas las tablas
    return cols.join(dtyp).join(misval).join(presval).join(unival).join(minval).join(maxval).join(mean).join(Std).join(Var).join(median).join(skewness).join(kurtosis)

#%% Obtaining the data quality report
report = dqr(data)

#%% Use of Seaborn library
import seaborn as sns
# Option 1. "all-vs-all", individual histogram

sns.pairplot(data) 
plt.show()

# Option 2. Separation of distributions based on output variable
sns.pairplot(data,hue='Refractive_index')
plt.show()
# Option 3. Select variables to plot "all-vs-all"
sns.pairplot(data,vars=['Refractive_index','Na','Mg'])
plt.show()
# Option 4. Selection of variables of interest
fig = sns.pairplot(data,x_vars=['Na','Mg'], y_vars=['Refractive_index'])
plt.show()

#%% View one of the variables
fig = plt.figure(figsize=(10,5))
plt.scatter(data.ID,data.Refractive_index)
plt.xlabel('ID'),plt.ylabel('Refractive index')
plt.grid()
plt.show()
# fig.savefig('../figures/P1_fig/F1.png')

#%% View one of the variables
fig = plt.figure(figsize=(5,4))
plt.scatter(data.ID,data.Na)
plt.xlabel('ID'),plt.ylabel('Sodio (Na)')
plt.grid()
plt.show()
# fig.savefig('../figures/P1_fig/F2.png')

#%% View one of the variables
fig = plt.figure(figsize=(5,4))
plt.scatter(data.ID,data.Na)
plt.xlabel('ID'),plt.ylabel('Ba')
plt.grid()
plt.show()
# fig.savefig('../figures/P1_fig/F2.png')

#%%  View one of the variables
fig = plt.figure(figsize=(5,4))
plt.scatter(data.Refractive_index,data.Na)
plt.xlabel('Refractive index'),plt.ylabel('Sodio (Na)')
# plt.axis('square')
plt.grid()
plt.show()
# fig.savefig('../figures/P1_fig/F3.png')

#%%

for n in names:
    fig = plt.figure(figsize=(10,5))
    plt.scatter(data.ID, data[n])
    plt.xlabel("ID"),plt.ylabel(n)
    plt.title(n)
    plt.grid()
    plt.show()

#%%
for n in names:
    fig = plt.figure(figsize=(10,5))
    plt.scatter(data.Refractive_index,data[n])
    plt.xlabel('Refractive index'),plt.ylabel(n)
    plt.title(n)
    plt.grid()
    plt.show()

#%% SCALING OF VARIABLES BY NORMALIZATION
data['Refractive_index_scale'] = (data.Refractive_index-data.Refractive_index.mean())/data.Refractive_index.std()
data['Na_scale'] = (data.Na-data.Na.mean())/data.Na.std()

### Scaling through scikit-learn
from sklearn import preprocessing
data['Refractive_index_scale'] = preprocessing.scale(data.Refractive_index)
data['Na_scale'] = preprocessing.scale(data.Na)


#%% Display the new variable
fig = plt.figure()
plt.subplot(1,2,1)
plt.scatter(data.ID,data.Na)
plt.xlabel('ID'),plt.ylabel('Sodio (Na)')
plt.title('Original')
plt.grid()
plt.subplot(1,2,2)
plt.scatter(data.ID,data.Na_scale)
plt.xlabel('ID'),plt.ylabel('Na_scale')
plt.title('Rescaled')
plt.grid()
fig.tight_layout()
plt.show()
# fig.savefig('../figures/P1_fig/F4.png')

#%% Display the new variable
fig = plt.figure()
plt.subplot(1,2,1)
plt.scatter(data.Refractive_index,data.Na)
plt.xlabel('Refractive index'),plt.ylabel('Sodio (Na)')
plt.axis('square')
plt.title('Original')
plt.grid()
plt.subplot(1,2,2)
plt.scatter(data.Refractive_index_scale,data.Na_scale)
plt.xlabel('Refractive index scale'),plt.ylabel('Sodio (Na) scale')
plt.axis('square')  
plt.title('Rescaled')
plt.grid()
fig.tight_layout()
plt.show()
# fig.savefig('../figures/P1_fig/F5.png')

#%% END OF CLASS 

#%% ASYMMETRY IN THE VARIABLES
fig = plt.figure()
plt.hist(data.Refractive_index,bins=30)
plt.xlabel('Refractive_index'),plt.ylabel('Frequency')
plt.vlines(data.Refractive_index.mean(),0,50,'r')
plt.show()
# fig.savefig('../figures/P1_fig/F6.png')


#%% Empirical criterion to consider that the data may have asymmetry
ratio = data.max()/(data.min()+1)

#%% Calculation of skewness
v = np.sum(np.power(data-data.mean(axis=0),2))/(data.shape[0]-1)
skewness = np.sum(np.power(data-data.mean(axis=0),3))/((data.shape[0]-1)*np.power(v,3/2))

## Calculation of skewness with pandas
skewness = data.skew()
#kurtosis = data.kurtosis()

## Calculation of skewness with scipy
from scipy import stats
skewness = stats.skew(data)


## Calculation of Kurtosis with scipy
from scipy import stats
kurtosis = stats.kurtosis(data)


#%% Skewness verification by means of histograms
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Refractive_index)
plt.xlabel('Refractive_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Mg)
plt.xlabel('Magnesium (Mg)'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()

#%% Transformation to limit skewness
data['Refractive_index_no_skewness'] = np.sqrt(data.Refractive_index)


fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Refractive_index)
plt.xlabel('Refractive_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Refractive_index_no_skewness)
plt.xlabel('Refractive_index_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()

#%%
data['Refractive_index_no_skewness'] = np.log(data.Refractive_index)
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Refractive_index)
plt.xlabel('Refractive_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Refractive_index_no_skewness)
plt.xlabel('Refractive_index_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()

#%%
data['Refractive_index_no_skewness'] = 1/data.Refractive_index

fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Refractive_index)
plt.xlabel('Refractive_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Refractive_index_no_skewness)
plt.xlabel('Refractive_index_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()
#%%
### BoxCox transformation using scipy
from scipy import stats
data['Refractive_index_no_skewness'] = stats.boxcox(data.Refractive_index,lmbda=-5.99911445243185)
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Refractive_index)
plt.xlabel('Refractive_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Refractive_index_no_skewness)
plt.xlabel('Refractive_index_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()
#%%
data['Refractive_index_no_skewness'],lamb = stats.boxcox(data.Refractive_index)


#%% Transformation to limit skewness
data['Mg_index_no_skewness'] = np.sqrt(data.Mg)


fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Mg)
plt.xlabel('Mg_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Mg_index_no_skewness)
plt.xlabel('Mg_index_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()

#%
data['Mg_index_no_skewness'] = np.log(data.Mg+1)
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Mg)
plt.xlabel('Mg_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Mg_index_no_skewness)
plt.xlabel('Mg_index_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()

#%
data['Mg_index_no_skewness'] = 1/(data.Mg)

fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Mg)
plt.xlabel('Mg_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Mg_index_no_skewness)
plt.xlabel('Mg_index_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()
#%
### BoxCox transformation using scipy
from scipy import stats
# data['Mg_index_no_skewness'] = stats.boxcox(data.Mg+1,lmbda=2.23672519042031)
data['Mg_index_no_skewness'] = stats.yeojohnson(data.Mg,lmbda=2.23672519042031)  # Box-Cox sólo funciona con números positivos, hay que usar el Box-Cox mejorado que es Yeo-Johnson
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Mg)
plt.xlabel('Mg_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Mg_index_no_skewness)
plt.xlabel('Mg_index_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()
#%
# data['Mg_index_no_skewness'],lamb = stats.boxcox(data.Mg+1)
data['Mg_index_no_skewness'],lamb = stats.yeojohnson(data.Mg)

#%%%%% ASYMMETRY IN THE VARIABLES
fig = plt.figure()
plt.hist(data.K,bins=30)
plt.xlabel('K'),plt.ylabel('Frequency')
plt.vlines(data.K.mean(),0,50,'r')
plt.show()
# fig.savefig('../figures/P1_fig/F6.png')


#%% Empirical criterion to consider that the data may have asymmetry
ratio = data.max()/data.min()

#%% Calculation of skewness
v = np.sum(np.power(data-data.mean(axis=0),2))/(data.shape[0]-1)
skewness = np.sum(np.power(data-data.mean(axis=0),3))/((data.shape[0]-1)*np.power(v,3/2))

## Calculation of skewness with pandas
#skewness = data.skew()
#kurtosis = data.kurtosis()

## Calculation of skewness with scipy
#from scipy import stats
#skewness = stats.skew(data)


## Calculation of Kurtosis with scipy
from scipy import stats
kurtosis = stats.kurtosis(data)


#%% Skewness verification by means of histograms
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.K)
plt.xlabel('K'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Mg)
plt.xlabel('Magnesium (Mg)'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()

#%% Transformation to limit skewness
data['K_no_skewness'] = np.sqrt(data.K)


fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.K)
plt.xlabel('K'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.K_no_skewness)
plt.xlabel('K_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()

#%%
data['Refractive_index_no_skewness'] = np.log(data.Refractive_index)
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Refractive_index)
plt.xlabel('Refractive_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Refractive_index_no_skewness)
plt.xlabel('Refractive_index_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()

#%%
data['Refractive_index_no_skewness'] = 1/data.Refractive_index

fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Refractive_index)
plt.xlabel('Refractive_index'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.Refractive_index_no_skewness)
plt.xlabel('Refractive_index_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()
#%%
### BoxCox transformation using scipy
from scipy import stats
data['K_no_skewness'] = stats.boxcox(data.K+0.001,lmbda=lamb)
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.K)
plt.xlabel('K'),plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(data.K_no_skewness)
plt.xlabel('K_no_skewness'),plt.ylabel('Frequency')
fig.tight_layout()
plt.show()
#%%
data['K_no_skewness'],lamb = stats.boxcox(data.K+0.001)



#%% Skewness check
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(data.Refractive_index)
plt.xlabel('Refractive_index'),plt.ylabel('Frequency')
plt.title('Skewness: %0.3f'%data['Refractive_index'].skew())
plt.subplot(1,2,2)
plt.hist(data.Refractive_index_no_skewness)
plt.xlabel('Refractive_index_no_skewness'),plt.ylabel('Frequency')
plt.title('Skewness: %0.3f'%data['Refractive_index_no_skewness'].skew())
fig.tight_layout()
plt.show()
# fig.savefig('../figures/P1_fig/F7.png')






#%% MAKING USE OF PANDAS PROFILING

summary = data.describe()


##Anaconda
# conda install -c conda-forge pandas-profiling
#pip install ydata-profiling
## Other
#pip install -U pandas-profiling[notebook]
#jupyter nbextension enable --py widgetsnbextension




import ydata_profiling
#report = data.profile_report()
report = ydata_profiling.ProfileReport(data)
report.to_file(output_file="Glass data profiling2.html")


# %%
