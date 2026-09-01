import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

# Function to determine the outliers using IQR
def find_boundaries(df, distance=1.5):
    IQR = df.quantile(0.75) - df.quantile(0.25)
    lower = df.quantile(0.25) - IQR * distance
    upper = df.quantile(0.75) + IQR * distance
    return lower, upper

# Z-Score Method
def find_outliers_zscore(df, threshold=3):
    z_scores = (df - df.mean()) / df.std()
    return np.abs(z_scores) > threshold

# Modified Z-Score Method
def find_outliers_modified_zscore(df, threshold=3.5):
    median = df.median()
    median_absolute_deviation = np.median(np.abs(df - median))
    modified_z_scores = 0.6745 * (df - median) / median_absolute_deviation
    return np.abs(modified_z_scores) > threshold


# Mahalanobis Distance Function
def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    inv_cov = np.linalg.inv(cov)
    md = np.sqrt(diff.dot(inv_cov).dot(diff.T))
    return md

def detect_outliers_mahalanobis(df, significance_level=0.01):
    mean = df.mean()
    cov = df.cov()
    inv_cov = np.linalg.inv(cov)
    mahalanobis_dist = df.apply(lambda x: mahalanobis_distance(x, mean, inv_cov), axis=1)
    threshold = chi2.ppf(1 - significance_level, df=df.shape[1]) ** 0.5
    outliers = mahalanobis_dist > threshold
    return outliers

# Function to create boxplot
def create_boxplot(data, title):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data)
    plt.title(title)
    plt.show()
    
def plot_scatter_with_outliers(data, outliers_indices, x_col, y_col):
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data[x_col], data[y_col], c='blue', label='Data')
    plt.scatter(data.loc[outliers_indices, x_col], data.loc[outliers_indices, y_col], c='red', label='Outliers')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    plt.legend()
    plt.show()
#%%
# Import the dataset
data = pd.read_csv('../Data/glass/glass.data', header=None, names=['ID','Refractive_index','Na','Mg','Al','Si','K','Ca','Ba','Fe','Glass_type'])

# Before removing outliers
create_boxplot(data['Refractive_index'], "Boxplot - Before Removing Outliers (Original Data)")

# IQR Method
lmin, lmax = find_boundaries(data['Refractive_index'])
outliers_iqr = np.where((data['Refractive_index'] < lmin) | (data['Refractive_index'] > lmax), True, False)
outliers_iqr_indices = data.index[outliers_iqr]
data_iqr_removed = data.drop(outliers_iqr_indices)
create_boxplot(data_iqr_removed['Refractive_index'], "Boxplot - After Removing Outliers (IQR Method)")
plot_scatter_with_outliers(data, outliers_iqr_indices, 'Refractive_index', 'Na')



# Z-Score Method
outliers_zscore = find_outliers_zscore(data['Refractive_index'])
outliers_zscore_indices = data.index[outliers_zscore]
data_zscore_removed = data.drop(outliers_zscore_indices)
create_boxplot(data_zscore_removed['Refractive_index'], "Boxplot - After Removing Outliers (Z-Score Method)")
plot_scatter_with_outliers(data, outliers_zscore_indices, 'Refractive_index', 'Na')


# Modified Z-Score Method
outliers_modified_zscore = find_outliers_modified_zscore(data['Refractive_index'])
outliers_modified_zscore_indices = data.index[outliers_modified_zscore]
data_modified_zscore_removed = data.drop(outliers_modified_zscore_indices)
create_boxplot(data_modified_zscore_removed['Refractive_index'], "Boxplot - After Removing Outliers (Modified Z-Score Method)")
plot_scatter_with_outliers(data, outliers_modified_zscore_indices, 'Refractive_index', 'Na')

# Mahalanobis Distance Method
outliers_mahalanobis = detect_outliers_mahalanobis(data.drop(columns=['ID', 'Glass_type']))
outliers_mahalanobis_indices = data.index[outliers_mahalanobis]
data_mahalanobis_removed = data.drop(outliers_mahalanobis_indices)
create_boxplot(data_mahalanobis_removed['Refractive_index'], "Boxplot - After Removing Outliers (Mahalanobis Distance Method)")
plot_scatter_with_outliers(data, outliers_mahalanobis_indices, 'Refractive_index', 'Na')


#%%
