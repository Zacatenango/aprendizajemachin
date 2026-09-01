# -*- coding: utf-8 -*-

# Import libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

#%% Import the dataset
data = pd.read_csv('glass.data', header=None)
names = ['ID', 'Refractive_index', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Glass_type']
data.columns = names

#%% Select relevant columns
data = data[[ 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]




#%% Variance Criteria for Variable Elimination
variances = pd.DataFrame(data.var().sort_values(), columns=['Variance'])
fig = plt.figure(figsize=(10, 8))
plt.bar(np.arange(len(variances)), variances.Variance)
plt.ylabel('Variance')
plt.xticks(np.arange(len(variances)), variances.index, rotation=90)
plt.tight_layout()
plt.show()

# Feature Selection
sel = VarianceThreshold(threshold=0.75)
data_clean = sel.fit_transform(data)

# Get selected feature indices
selected_indices = sel.get_support(indices=True)

# Get selected feature names
selected_feature_names = data.columns[selected_indices]

# Convert data_clean to DataFrame with selected column names
data_clean = pd.DataFrame(data_clean, columns=selected_feature_names)

#%% Correlation analysis for elimination of variables
import matplotlib.pyplot as plt
correlations = np.corrcoef(data,rowvar=False)

fig = plt.figure()
plt.imshow(correlations)
plt.xticks(np.arange(len(data.columns)), data.columns)
plt.yticks(np.arange(len(data.columns)), data.columns)
plt.colorbar()
plt.show()
# fig.savefig('../figures/P1_fig/F10.png')


#%% Variance Influence Factor

import statsmodels.api as sm
import matplotlib.pyplot as plt

# Calculate VIF for each feature
def calculate_vif(data):
    features = data.columns
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data['VIF'] = [1 / (1 - sm.OLS(data[col], data.drop(columns=[col])).fit().rsquared)  for col in data.columns ]
    return vif_data

# Calculate VIF for original data
vif_scores_before = calculate_vif(data)

# Set threshold for VIF  #VIF (Variance Inflation Factor) > 10
vif_threshold  =10

# Select variables based on VIF
selected_features_vif = vif_scores_before[vif_scores_before['VIF'] < vif_threshold]['Feature'].tolist()

# Subset the original data based on selected features
data_vif_selected = data[selected_features_vif]

# Calculate VIF after selection
vif_scores_after = calculate_vif(data_vif_selected)

# Plot VIF before and after
max_vif = max(vif_scores_before['VIF'].max(), vif_scores_after['VIF'].max(), 10)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].bar(vif_scores_before['Feature'], vif_scores_before['VIF'])
axes[0].set_title('VIF Before Selection')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('VIF')
axes[0].tick_params(axis='x', rotation=90)
axes[0].axhline(y=vif_threshold, color='red', linestyle='--')  #
axes[0].set_ylim(0, max_vif)  # Set y-axis limit
axes[1].bar(vif_scores_after['Feature'], vif_scores_after['VIF'])
axes[1].set_title('VIF After Selection')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('VIF')
axes[1].tick_params(axis='x', rotation=90)
axes[1].axhline(y=vif_threshold, color='red', linestyle='--')
axes[1].set_ylim(0, max_vif)  
plt.tight_layout()
plt.show()




#%% Hierarchical clustering application
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Perform hierarchical clustering
Z = linkage(data.T, metric='correlation', method='complete')

# Visualize the dendrogram with feature names
d = dendrogram(Z, labels=data.columns)
plt.xticks(rotation=90)
plt.show()


# Feature selection based on hierarchical clustering
# For example, selecting the first feature from each cluster
selected_features_clust = [d['leaves'][i] for i in range(len(d['leaves'])) if i % 5 == 0]

# Subset the original data based on selected features
data_clust_selected = data.iloc[:, selected_features_clust]



# Perform hierarchical clustering
Z = linkage(data_clust_selected.T, metric='correlation', method='complete')

# Visualize the dendrogram with feature names
d = dendrogram(Z, labels=data_clust_selected.columns)
plt.xticks(rotation=90)
plt.show()

