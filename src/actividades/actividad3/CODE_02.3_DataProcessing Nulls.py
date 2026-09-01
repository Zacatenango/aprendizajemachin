# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:08:02 2024

@author: zaratejo
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor

# Import the table
datamovie = pd.read_excel('Test de películas (anónimo)(1-12).xlsx')

# Selection of valid columns
csel = np.arange(19, 247, 3)
cnames = list(datamovie.columns.values[csel])
datan = datamovie[cnames]
#%%
# Delete all records with null data
datan_clean_dropna = datan.dropna()

# Imputation by mean or median
imputer_mean = SimpleImputer(strategy='mean')
datan_clean_mean = pd.DataFrame(imputer_mean.fit_transform(datan), columns=datan.columns)

imputer_median = SimpleImputer(strategy='median')
datan_clean_median = pd.DataFrame(imputer_median.fit_transform(datan), columns=datan.columns)

imputer_mode = SimpleImputer(strategy='most_frequent')
datan_clean_mode = pd.DataFrame(imputer_mode.fit_transform(datan), columns=datan.columns)

# Iterative imputation
imputer_bayesian = IterativeImputer(estimator=BayesianRidge(), max_iter=100, random_state=0)
datan_clean_bayesian = pd.DataFrame(imputer_bayesian.fit_transform(datan), columns=datan.columns)

imputer_knn = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=2), max_iter=10, random_state=0)
datan_clean_knn = pd.DataFrame(imputer_knn.fit_transform(datan), columns=datan.columns)

# Forward fill
datan_clean_ffill = datan.ffill()

# Backward fill
datan_clean_bfill = datan.bfill()

# Interpolation
# La interpolación es la que más respeta el comportamiento de los datos
datan_clean_interp = datan.interpolate()

# Custom imputation (filling missing values with a constant)
datan_clean_custom = datan.fillna(value=0)  # Replace NaN with 0

# Custom imputation (filling missing values with a specific value based on domain knowledge)
datan_clean_domain = datan.fillna(value=-1)

