# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:44:18 2025

@author: zaratejo
"""
# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, median_absolute_error)

#%%Datasets 
x_common = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
# Dataset 1
x1 = x_common.copy()
y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
# Dataset 2
x2 = x_common.copy()
y2 = np.array([ 9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
# Dataset 3
x3 = x_common.copy()
y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
# Dataset 4
x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float)
y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04,5.25, 12.50, 5.56, 7.91, 6.89])

# Merge datasets & Plot
datasets = {
    "Dataset 1: Linear": (x1, y1),
    "Dataset 2: Nonlinear": (x2, y2),
    "Dataset 3: Outlier": (x3, y3),
    "Dataset 4: High leverage": (x4, y4)}

fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(12, 8), sharex=True,sharey=True)
axes = axes.flatten()

for index, dataset_name in enumerate(datasets):
    x = datasets[dataset_name][0]
    y = datasets[dataset_name][1]
    ax = axes[index]
    ax.scatter( x,  y, s=90, color="steelblue", edgecolor="black", alpha=0.9 )
    ax.set_title(dataset_name, fontsize=12,fontweight="bold" )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(3, 20)
    ax.set_ylim(3, 14)
    ax.grid(alpha=0.25)
# Tittle
fig.suptitle("Anscombe's Quartet dataset", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%% linear regression model

models = {}
predictions = {}
residuals = {}
results = {}


for dataset_name in datasets:
    x = datasets[dataset_name][0]
    y = datasets[dataset_name][1]
    x_model = x.reshape(-1, 1)
    # Regression model
    model = LinearRegression()
    model.fit(x_model, y)
    y_predicted = model.predict(x_model)
    # Calculate residuals
    dataset_residuals = y - y_predicted
    
    x_mean = np.mean(x)
    x_variance = np.var(x, ddof=1)

    y_mean = np.mean(y)
    y_variance = np.var(y, ddof=1)
 
    #equation
    correlation = np.corrcoef(x, y)[0, 1]
    intercept = model.intercept_
    slope = model.coef_[0]
    #R2
    r2 = r2_score(y, y_predicted)
    #RMSE
    rmse = np.sqrt(mean_squared_error(y, y_predicted))
    #MAE
    mae = mean_absolute_error(y, y_predicted)
    #MedianAE
    median_ae = median_absolute_error(y,y_predicted)
    #residuals
    models[dataset_name] = model
    predictions[dataset_name] = y_predicted
    residuals[dataset_name] = dataset_residuals
    results[dataset_name] = {
        "X mean": x_mean,
        "X variance": x_variance,
        "Y mean": y_mean,
        "Y variance": y_variance,
        "Correlation": correlation,
        "Intercept": intercept,
        "Slope": slope,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Median Absolute Error": median_ae
    }


# 
#%%Plot Linnear regression
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(14, 10),sharex=True, sharey=True)
axes = axes.flatten()

for index, dataset_name in enumerate(datasets):
    x = datasets[dataset_name][0]
    y = datasets[dataset_name][1]
    model = models[dataset_name]
    dataset_results = results[dataset_name]
    ax = axes[index]
    ax.scatter(x,y,s=90, color="steelblue", edgecolor="black", alpha=0.9, zorder=3 )

    line_x = np.linspace(3, 20, 200)
    line_x_model = line_x.reshape(-1, 1)
    line_y = model.predict(line_x_model)
    # Plot the regression line
    ax.plot(line_x,line_y, color="darkred", linewidth=2,label="Linear regression", zorder=2 )

    # Metrics in text box 1
    statistics_text = (
        f"Mean X = {dataset_results['X mean']:.3f}\n"
        f"Variance X = {dataset_results['X variance']:.3f}\n"
        f"Mean Y = {dataset_results['Y mean']:.3f}\n"
        f"Variance Y = {dataset_results['Y variance']:.3f}"
    )
    ax.text(0.03, 0.95,statistics_text,transform=ax.transAxes, verticalalignment="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8,"edgecolor": "gray" })
    # Equation box 2
    regression_text = (
        f"Y = {dataset_results['Intercept']:.3f} "
        f"+ {dataset_results['Slope']:.3f}X\n"
        f"r = {dataset_results['Correlation']:.3f}\n"
        f"R² = {dataset_results['R2']:.3f}"
    )

    ax.text(0.97, 0.95,regression_text, transform=ax.transAxes,horizontalalignment="right", 
            verticalalignment="top",  fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray" }
    )

    #Format & titles 
    ax.set_title(dataset_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(3, 20)
    ax.set_ylim(3, 14)
    ax.grid(alpha=0.25)
    ax.legend( loc="lower right",fontsize=8 )


# Title 
fig.suptitle("Anscombe's Quartet: Same Regression, Different Patterns", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%% Actual Y vs Predicted Y
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()
for index, dataset_name in enumerate(datasets):
    ax = axes[index]
    y_actual = datasets[dataset_name][1]
    predicted = predictions[dataset_name]
    ax.scatter(y_actual,y_predicted,s=90,color="lightblue", edgecolor="black",alpha=0.9)
    # 45° line = perfect prediction
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val],[min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Fit"   )
    # Add metrics
    ax.text( 0.05,0.95,
        f"R² = {results[dataset_name]['R2']:.3f}\n"
        f"RMSE = {results[dataset_name]['RMSE']:.3f}\n"
        f"MAE = {results[dataset_name]['MAE']:.3f}",
        transform=ax.transAxes, verticalalignment="top",
        bbox=dict(facecolor="white",alpha=0.8,edgecolor="gray" ))
    ax.set_title(dataset_name,fontsize=12,fontweight="bold" )
    ax.set_xlabel("Actual Y")
    ax.set_ylabel("Predicted Y")
    ax.grid(alpha=0.25)
    ax.legend()

fig.suptitle("Actual vs Predicted Values", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


#%% Metrics comparison table
metrics_rows = []
for dataset_name in datasets:
    dataset_results = results[dataset_name]
    metrics_rows.append({
        "Dataset": dataset_name,
        "Correlation": dataset_results["Correlation"],
        "R2": dataset_results["R2"],
        "RMSE": dataset_results["RMSE"],
        "MAE": dataset_results["MAE"],
        "Median Absolute Error":
            dataset_results["Median Absolute Error"]
    })

metrics_table = pd.DataFrame(metrics_rows)
metrics_table = metrics_table.round(3)

print("\nMetrics")
print("===================")
print(metrics_table.to_string(index=False))


#%% Plot the residuals

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12, 8), sharex=True)
axes = axes.flatten()
for index, dataset_name in enumerate(datasets):
    ax = axes[index]
    # Extract predictions and residuals
    y_predicted = predictions[dataset_name]
    dataset_residuals = residuals[dataset_name]
    # Plot predicted values against residuals
    ax.scatter(y_predicted, dataset_residuals, s=90,color="darkorange", edgecolor="black", alpha=0.9)
    # Reference line =0
    ax.axhline( y=0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(dataset_name, fontsize=12,fontweight="bold" )
    ax.set_xlabel("Predicted Y")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.25)

fig.suptitle("Residual Analysis", fontsize=16,fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


#%% Boxplot 
fig, axes = plt.subplots(nrows=2,  ncols=2,figsize=(12, 8), sharey=True)
axes = axes.flatten()
for index, dataset_name in enumerate(datasets):
    ax = axes[index]
    dataset_residuals = residuals[dataset_name]
    ax.boxplot(dataset_residuals,patch_artist=True, boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red", linewidth=2))
    # Reference line =0
    ax.axhline(y=0,color="black", linestyle="--",linewidth=1)
    ax.set_title(dataset_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.25)

fig.suptitle("Residual Boxplots",fontsize=16,fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%% Histograms of residuals
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()
for index, dataset_name in enumerate(datasets):
    ax = axes[index]
    dataset_residuals = residuals[dataset_name]
    ax.hist(dataset_residuals, bins=6,color="lightblue", edgecolor="black", alpha=0.8)
    # Reference line =0
    ax.axvline(x=0,color="black", linestyle="--",linewidth=2)
    ax.set_title( dataset_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)

fig.suptitle("Residual Histograms",fontsize=16,fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%% 
# 10. Print the main interpretation
# 

print("\nInterpretation")
print("==============")

print(
    """
Dataset 1:
The data follows a reasonably linear pattern.
RMSE is an appropriate primary performance metric.

Dataset 2:
The data contains a curved relationship.
The main issue is model specification, not the metric.
A polynomial or nonlinear model should be evaluated.

Dataset 3:
A single large outlier affects the regression results.
MAE or Median Absolute Error is more robust than RMSE.

Dataset 4:
One high-leverage observation determines the regression line.
Cross-validation and influence diagnostics are required.

General recommendation:
Use MAE as a general, interpretable performance metric.
Use R-squared only as a secondary descriptive metric.
Always inspect the raw data and residual plots.
"""
)