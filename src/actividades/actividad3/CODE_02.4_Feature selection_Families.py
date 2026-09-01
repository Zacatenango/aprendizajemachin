import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectKBest, chi2, RFE, VarianceThreshold, mutual_info_classif, SequentialFeatureSelector, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
data = load_diabetes()
X, y = data.data, data.target

# Ensure all features are non-negative for chi2 and mutual_info_classif
X_non_negative = X - X.min(axis=0)

# Get feature names
feature_names = data.feature_names

# Initialize a DataFrame to store the results
results_df = pd.DataFrame(index=feature_names)

#%% Filter Methods: Evaluate features independently of the model

# Variance Criteria: Remove features with low variance
threshold = 0.0  # Adjusted threshold for variance to avoid errors
variance_selector = VarianceThreshold(threshold=threshold)
X_variance = variance_selector.fit_transform(X)
variance_features = variance_selector.get_support(indices=True)
results_df['Variance Criteria'] = ['Keep' if i in variance_features else 'Remove' for i in range(len(feature_names))]

# SelectKBest with chi-square test: Select features based on chi-square test
select_k_best = SelectKBest(score_func=chi2, k=5)
X_k_best = select_k_best.fit_transform(X_non_negative, y)
k_best_features = select_k_best.get_support(indices=True)
results_df['SelectKBest'] = ['Keep' if i in k_best_features else 'Remove' for i in range(len(feature_names))]

# Information Gain: Calculate the reduction in entropy from the transformation of a dataset
info_gain_selector = SelectKBest(score_func=mutual_info_classif, k=5)
X_info_gain = info_gain_selector.fit_transform(X, y)
info_gain_features = info_gain_selector.get_support(indices=True)
results_df['Information Gain'] = ['Keep' if i in info_gain_features else 'Remove' for i in range(len(feature_names))]

# Correlation Coefficient: Measure the linear relationship between features and the target variable
X_df = pd.DataFrame(X, columns=feature_names)  # Define X_df before use
correlations = X_df.corrwith(pd.Series(y)).apply(abs).sort_values(ascending=False)
# Keep top 5 features
correlation_threshold =correlations.iloc[4] #0.5#correlations.iloc if len(correlations) > 4 else correlations.iloc[-1] # correlations.iloc 
correlation_features = correlations[correlations >= correlation_threshold].index.tolist()


correlation_features = correlations[correlations >= correlation_threshold].index.tolist()
results_df['Correlation Coefficient'] = ['Keep' if feature_names[i] in correlation_features else 'Remove' for i in range(len(feature_names))]

#%% Wrapper Methods: Evaluate feature subsets by training and testing a model on different combinations of features

# Recursive Feature Elimination (RFE): Remove features recursively based on their importance to the model
model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
rfe_features = rfe.get_support(indices=True)
results_df['RFE'] = ['Keep' if i in rfe_features else 'Remove' for i in range(len(feature_names))]

# Forward Selection: Add features one by one based on their contribution to model performance
forward_selector = SequentialFeatureSelector(model, n_features_to_select=5, direction='forward')
X_forward = forward_selector.fit_transform(X, y)
forward_features = forward_selector.get_support(indices=True)
results_df['Forward Selection'] = ['Keep' if i in forward_features else 'Remove' for i in range(len(feature_names))]

# Backward Elimination: Start with all features and remove them one by one based on their contribution to model performance
backward_selector = SequentialFeatureSelector(model, n_features_to_select=5, direction='backward')
X_backward = backward_selector.fit_transform(X, y)
backward_features = backward_selector.get_support(indices=True)
results_df['Backward Elimination'] = ['Keep' if i in backward_features else 'Remove' for i in range(len(feature_names))]

#%% Embedded Methods: Perform feature selection during the model training process

# LASSO (Least Absolute Shrinkage and Selection Operator): Select features using LASSO regularization
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
lasso_model = SelectFromModel(lasso, prefit=True)
X_lasso = lasso_model.transform(X)
lasso_features = lasso_model.get_support(indices=True)
results_df['LASSO'] = ['Keep' if i in lasso_features else 'Remove' for i in range(len(feature_names))]

# Elastic Net Regularization: Combine LASSO and Ridge regularization to select features
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net_model = SelectFromModel(elastic_net, prefit=True)
X_elastic_net = elastic_net_model.transform(X)
elastic_net_features = elastic_net_model.get_support(indices=True)
results_df['Elastic Net'] = ['Keep' if i in elastic_net_features else 'Remove' for i in range(len(feature_names))]

# Principal Component Analysis (PCA): Transform features into principal components based on their variance
n_components = 5  # Number of principal components to keep
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
pca_explained_variance_ratio = pca.explained_variance_ratio_
pca_threshold_index = pca_explained_variance_ratio.argsort()[-n_components:]
pca_features = [feature_names[i] for i in pca_threshold_index]
results_df['PCA'] = ['Keep' if feature_names[i] in pca_features else 'Remove' for i in range(len(feature_names))]

# Decision Tree Feature Importance: Provide feature importance scores based on information gain
tree = DecisionTreeRegressor()
tree.fit(X, y)
tree_importances_threshold_index = tree.feature_importances_.argsort()[-5:]
tree_features = [feature_names[i] for i in tree_importances_threshold_index]
results_df['Decision Tree'] = ['Keep' if feature_names[i] in tree_features else 'Remove' for i in range(len(feature_names))]




# Filter Method: Variance Inflation Factor (VIF): Measure the multicollinearity of features
vif_data = pd.DataFrame()  # Define vif_data before use
vif_data["feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
vif_threshold_index = vif_data["VIF"].argsort()[-5:]
vif_features = [feature_names[i] for i in vif_threshold_index]
results_df['VIF'] = ['Keep' if feature_names[i] in vif_features else 'Remove' for i in range(len(feature_names))]

print(results_df)
# Save the results to a CSV file
#results_df.to_csv("feature_selection_results.csv")
#print("The feature selection results have been saved to feature_selection_results.csv.")