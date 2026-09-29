# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:30:49 2024

@author: zaratejo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold

# Set random seed for reproducibility
np.random.seed(0)

# Generate a simulated dataset
X = np.linspace(0, 10, 100)  # Feature values
y = 2 * X + np.random.normal(0, 1, 100)  # Target with noise

# Convert to DataFrame for easy plotting
df = pd.DataFrame({'X': X, 'y': y})

# Plot all data in one chart
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='X', y='y')
plt.title('All Data')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend(title='Dataset')
plt.show()

#%%

# Split the dataset into training (60%), validation (20%), and test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
#X_val1, X_test1, y_val1, y_test1 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

# Convert splits to DataFrames
df_train = pd.DataFrame({'X': X_train, 'y': y_train, 'Type': 'Training'})
df_val = pd.DataFrame({'X': X_val, 'y': y_val, 'Type': 'Validation'})
df_test = pd.DataFrame({'X': X_test, 'y': y_test, 'Type': 'Test'})

# Combine all DataFrames
df_all = pd.concat([df_train, df_val, df_test])



# Plot training, validation, and test data with different colors
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_train, x='X', y='y', color='blue', label='Training')
sns.scatterplot(data=df_val, x='X', y='y', color='orange', label='Validation')
sns.scatterplot(data=df_test, x='X', y='y', color='red', label='Test')
plt.title('Training, Validation, and Test Data')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend(title='Dataset')
plt.show()


#%%

# Define split value
split_value = 8

# Create a new test set starting from X = 8
df_test = df[df['X'] >= split_value]
df_train = df[df['X'] < split_value]

# Convert splits to DataFrames for training and test sets
df_train = df_train.copy()
df_train['Type'] = 'Training'
df_test = df_test.copy()
df_test['Type'] = 'Test'

# Combine all DataFrames
df_all = pd.concat([df_train, df_test])

# Plot all data in one chart
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_all, x='X', y='y', hue='Type', palette='deep')
plt.title('All Data with New Test Split')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend(title='Dataset')
plt.show()

# Plot the new test split
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_train, x='X', y='y', color='blue', label='Training')
sns.scatterplot(data=df_test, x='X', y='y', color='red', label='Test')
plt.axvline(x=split_value, color='grey', linestyle='--', label='Split Value (X=8)')
plt.title('Backtesting')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend(title='Dataset')
plt.show()

