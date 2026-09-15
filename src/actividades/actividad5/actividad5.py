# -*- coding: utf-8 -*-
'''
Actividad 5 - Linear regression vs PCA regression vs PLS regression
Dataset: UCI Automobile (https://archive.ics.uci.edu/dataset/10/automobile)

1) PROBLEM DESCRIPTION
----------------------
The Automobile dataset (1985 Ward's Automotive Yearbook) describes 205 cars by
their technical specifications (engine size, horsepower, curb weight,
dimensions, fuel consumption, ...), some categorical attributes (make, body
style, fuel type, ...) and an insurance risk rating ("symboling").

UCI lists "symboling" as the target, but it is an ordinal risk score. The
natural *regression* problem in this dataset is:

    Predict the market PRICE of a car from its numeric technical specifications.

This is a good case study for PCA and PLS: the predictors are highly
collinear (length, width, wheel-base, curb-weight and engine-size all measure
"how big the car is"; city-mpg and highway-mpg are ~0.97 correlated, etc.).
Collinearity inflates the variance of the ordinary least squares coefficients,
so projecting the inputs onto a few latent components can give a model that is
as accurate (or more) and much more stable:
    - PCA regression: components chosen to explain the variance of X only
      (unsupervised), then a LinearRegression on those components.
    - PLS regression: components chosen to maximize covariance between X and y
      (supervised), so usually fewer components are needed.

2) CLEAN-UP
-----------
    - Keep the numeric features (categorical ones are left out so PCA/PLS
      operate on a homogeneous, interpretable set of measurements).
    - Drop "normalized-losses" (~20% missing) and "symboling" (risk score, not
      a technical specification).
    - Remove rows with NA (only 12 rows, ~6%).
    - Outlier detection: univariate IQR (reported) and multivariate Mahalanobis
      distance (used for removal).
    - Train/test split 80/20.
    - Standard scaling fitted on the training set only (inside a Pipeline, so
      there is no leakage from the test set, also during cross-validation).
'''

#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
TARGET = 'price'

#%% Load data
automobile = fetch_ucirepo(id=10)
X = automobile.data.features
y = automobile.data.targets

print(automobile.metadata['abstract'])
print(automobile.variables[['name', 'role', 'type', 'missing_values']])

data = pd.concat([X, y], axis=1)
print('Original shape:', data.shape)

#%% Clean-up: select numeric features and remove NA
print('\nMissing values per column:')
print(data.isna().sum()[data.isna().sum() > 0])

data = data.drop(columns=['normalized-losses', 'symboling'])
data = data.select_dtypes(include='number')
data = data.dropna().reset_index(drop=True)
print('Shape after removing NA and non-numeric columns:', data.shape)

features = [c for c in data.columns if c != TARGET]

#%% Correlation between variables (motivation for PCA / PLS)
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation matrix - Automobile (numeric variables)')
plt.tight_layout()
plt.show()

#%% Outlier detection
# Univariate: IQR rule per column
def find_boundaries(df, distance=1.5):
    IQR = df.quantile(0.75) - df.quantile(0.25)
    lower = df.quantile(0.25) - IQR * distance
    upper = df.quantile(0.75) + IQR * distance
    return lower, upper

lower, upper = find_boundaries(data)
outliers_iqr = (data < lower) | (data > upper)
print('\nIQR outliers per column:')
print(outliers_iqr.sum())
print('Rows with at least one IQR outlier:', outliers_iqr.any(axis=1).sum())
# The IQR rule flags ~40% of the rows: num-of-cylinders is almost always 4,
# so every 5/6/8/12-cylinder car is flagged, and luxury cars are flagged by
# price. Those are real cars, not errors, so the per-column rule is too
# aggressive here. We use a multivariate criterion instead.

plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns))
plt.xticks(rotation=60)
plt.title('Boxplot of standardized variables - before removing outliers')
plt.tight_layout()
plt.show()

# Multivariate: Mahalanobis distance on the predictors
def detect_outliers_mahalanobis(df, significance_level=0.001):
    diff = (df - df.mean()).values
    inv_cov = np.linalg.inv(df.cov().values)
    md = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))
    threshold = np.sqrt(chi2.ppf(1 - significance_level, df=df.shape[1]))
    return pd.Series(md > threshold, index=df.index), md, threshold

outliers_md, md, md_threshold = detect_outliers_mahalanobis(data[features])
print('\nMahalanobis outliers (alpha=0.001):', outliers_md.sum())
print(data.loc[outliers_md, ['engine-size', 'horsepower', 'compression-ratio',
                             'num-of-cylinders', 'price']])

plt.figure(figsize=(10, 5))
plt.scatter(np.arange(len(md)), md, c=np.where(outliers_md, 'red', 'blue'))
plt.axhline(md_threshold, color='k', linestyle='--', label='chi2 threshold')
plt.xlabel('Observation'), plt.ylabel('Mahalanobis distance')
plt.title('Multivariate outliers (Mahalanobis distance)')
plt.legend()
plt.show()

data = data.loc[~outliers_md].reset_index(drop=True)
print('Shape after removing outliers:', data.shape)

#%% Train / test split
X = data[features]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=RANDOM_STATE)
print('Train:', X_train.shape, ' Test:', X_test.shape)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

def evaluate(name, model):
    y_predict = np.ravel(model.predict(X_test))
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_predict)),
        'MAE': mean_absolute_error(y_test, y_predict),
        'R2': r2_score(y_test, y_predict),
    }
    return y_predict

def plot_real_vs_predict(name, y_predict):
    ref = np.linspace(min(y_test), max(y_test))
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_predict)
    plt.plot(ref, ref, 'k--')
    plt.axis('square')
    plt.xlabel('price real'), plt.ylabel('price predict')
    plt.title('%s, RMSE=%0.1f, R^2=%0.4f' % (name, results[name]['RMSE'], results[name]['R2']))
    plt.grid()
    plt.show()

#%% Model 1: Linear regression (scaled original features)
linreg = Pipeline([
    ('scaler', StandardScaler()),
    ('linreg', LinearRegression()),
])
linreg.fit(X_train, y_train)
y_predict = evaluate('LinearRegression', linreg)
plot_real_vs_predict('LinearRegression', y_predict)

coef = pd.Series(linreg.named_steps['linreg'].coef_, index=features)
print('\nLinear regression coefficients (standardized X):')
print(coef.sort_values())
# Note the sign flips between highly correlated variables (e.g. city-mpg vs
# highway-mpg, length vs width): a symptom of collinearity.

#%% Model 2: PCA + Linear regression
# Explained variance on the (scaled) training data
pca_full = PCA().fit(StandardScaler().fit_transform(X_train))
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(cum_var) + 1), pca_full.explained_variance_ratio_, label='Individual')
plt.step(range(1, len(cum_var) + 1), cum_var, where='mid', color='r', label='Cumulative')
plt.axhline(0.95, color='k', linestyle='--', label='95%')
plt.xlabel('Principal component'), plt.ylabel('Explained variance ratio')
plt.title('PCA - explained variance (training set)')
plt.legend()
plt.show()
print('\nComponents needed for 95% of the variance:', np.argmax(cum_var >= 0.95) + 1)

# The number of components is chosen by cross-validation on the training set
pcr = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('linreg', LinearRegression()),
])
n_components = list(range(1, len(features) + 1))
pcr_search = GridSearchCV(pcr, {'pca__n_components': n_components},
                          cv=cv, scoring='neg_root_mean_squared_error')
pcr_search.fit(X_train, y_train)
best_pcr = pcr_search.best_estimator_
print('PCA regression - best n_components (CV):', pcr_search.best_params_['pca__n_components'])

name_pcr = 'PCA + LinearRegression (k=%d)' % pcr_search.best_params_['pca__n_components']
y_predict = evaluate(name_pcr, best_pcr)
plot_real_vs_predict(name_pcr, y_predict)

#%% Model 3: PLS regression
pls = Pipeline([
    ('scaler', StandardScaler()),
    ('pls', PLSRegression(scale=False)),
])
pls_search = GridSearchCV(pls, {'pls__n_components': n_components},
                          cv=cv, scoring='neg_root_mean_squared_error')
pls_search.fit(X_train, y_train)
best_pls = pls_search.best_estimator_
print('PLS regression - best n_components (CV):', pls_search.best_params_['pls__n_components'])

name_pls = 'PLSRegression (k=%d)' % pls_search.best_params_['pls__n_components']
y_predict = evaluate(name_pls, best_pls)
plot_real_vs_predict(name_pls, y_predict)

#%% Cross-validation error vs number of components (PCA vs PLS)
plt.figure(figsize=(9, 5))
plt.plot(n_components, -pcr_search.cv_results_['mean_test_score'], 'o-', label='PCA + LinearRegression')
plt.plot(n_components, -pls_search.cv_results_['mean_test_score'], 's-', label='PLSRegression')
plt.xlabel('Number of components'), plt.ylabel('CV RMSE')
plt.title('5-fold CV RMSE vs number of components (training set)')
plt.xticks(n_components)
plt.legend()
plt.grid()
plt.show()

#%% Scores of the first two components: PCA (unsupervised) vs PLS (supervised)
X_train_scaled = best_pls.named_steps['scaler'].transform(X_train)
pca_scores = PCA(n_components=2).fit_transform(X_train_scaled)
pls_scores = PLSRegression(n_components=2, scale=False).fit(X_train_scaled, y_train).transform(X_train_scaled)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, scores, title in zip(axes, [pca_scores, pls_scores], ['PCA', 'PLS']):
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=y_train, cmap='viridis')
    ax.set_xlabel('Component 1'), ax.set_ylabel('Component 2')
    ax.set_title('%s scores (color = price)' % title)
fig.colorbar(sc, ax=axes, label='price')
plt.show()

#%% Comparison of the three models on the test set
results_df = pd.DataFrame(results).T
results_df['CV RMSE (train)'] = [
    -cross_val_score(linreg, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error').mean(),
    -pcr_search.best_score_,
    -pls_search.best_score_,
]
print('\nTest-set comparison:')
print(results_df.round(4))

results_df[['RMSE', 'MAE']].plot(kind='bar', figsize=(9, 5), rot=0)
plt.title('Test error by model (lower is better)')
plt.ylabel('price units')
plt.tight_layout()
plt.show()

#%% Conclusions
'''
Results (random_state=42):

                                 Test RMSE   Test R2   CV RMSE (train)
    LinearRegression               ~3312      0.861        ~3324
    PCA + LinearRegression (k=7)   ~4063      0.791        ~3206
    PLSRegression (k=4)            ~3751      0.822        ~3207

- All three models explain ~80-86% of the price variance in the test set.
- PCA needs 8 components to keep 95% of the variance of X, and CV picks 7.
  PLS gets the same CV error with only 4 components, because its components
  are built using price (supervised), while PCA ignores the target.
- On cross-validation (5 folds over 148 cars), PCA and PLS are slightly better
  than plain linear regression. On the single 38-car test set plain linear
  regression comes out ahead. With so few observations the test ranking is
  noisy, so the CV error is the more reliable comparison.
- The main advantage of PCA/PLS here is stability and a simpler model: the
  OLS coefficients show sign flips between strongly correlated variables
  (e.g. city-mpg negative vs highway-mpg positive), which is typical of
  collinearity. PLS gives similar accuracy with a 4-dimensional representation.
- Note: the Mahalanobis filter removed some luxury cars (8/12-cylinder
  Jaguar/Mercedes). The models are therefore valid for the "regular" range of
  cars; very expensive cars would be extrapolation.
'''
