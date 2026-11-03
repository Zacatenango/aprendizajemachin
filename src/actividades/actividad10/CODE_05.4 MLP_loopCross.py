# -*- coding: utf-8 -*-
"""
Created on Mon May 19 22:23:46 2025

@author: zaratejo
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
df = sns.load_dataset('mpg')
print("Dataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values count:")
print(df.isnull().sum())

# Data Cleaning
# Drop rows with missing values
df_cleaned = df.dropna()
print("\nShape after dropping missing values:", df_cleaned.shape)

# Check data types and information
print("\nData types:")
print(df_cleaned.dtypes)

# Convert 'origin' to categorical if not already
if df_cleaned['origin'].dtype == 'object':
    df_cleaned['origin'] = pd.Categorical(df_cleaned['origin'])

# Define X and y
# For X, we'll use all features except 'mpg', 'name' and 'model_year' (to avoid data leakage)
X = df_cleaned.drop(['mpg', 'name', 'model_year'], axis=1)
y = df_cleaned['mpg']

print("\nFeatures (X):")
print(X.columns.tolist())
print("\nTarget (y): mpg")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Split data into training+validation (90%) and testing (10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Preprocess the data
X_train_val_processed = preprocessor.fit_transform(X_train_val)
X_test_processed = preprocessor.transform(X_test)

# Get the input shape after preprocessing (for the neural network)
input_shape = X_train_val_processed.shape[1]
print(f"\nInput shape after preprocessing: {input_shape}")

# Define the neural network configurations
configs = {
    'A': {
        'name': 'Config A: 3 hidden layers with 4 neurons each',
        'layers': [4, 4, 4]
    },
    'B': {
        'name': 'Config B: 1 hidden layer with 12 neurons',
        'layers': [12]
    },
    'C': {
        'name': 'Config C: 4 hidden layers with 3 neurons each',
        'layers': [3, 3, 3, 3]
    }
}

# Define optimizer configurations (not instances)
optimizer_configs = {
    'sgd': {'class': SGD, 'params': {'learning_rate': 0.01}},
    'adam': {'class': Adam, 'params': {'learning_rate': 0.001}},
    'rmsprop': {'class': RMSprop, 'params': {'learning_rate': 0.001}}
}

# Function to create model based on configuration
def create_model(config, optimizer_name, input_shape):
    model = Sequential()
    
    # Input layer
    model.add(Dense(config['layers'][0], activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    
    # Hidden layers
    for i in range(1, len(config['layers'])):
        model.add(Dense(config['layers'][i], activation='relu'))
        model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1))  # Linear activation for regression
    
    # Create a fresh optimizer instance
    opt_config = optimizer_configs[optimizer_name]
    optimizer = opt_config['class'](**opt_config['params'])
    
    # Compile model
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# Define K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Prepare for results
results = []

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train and evaluate each configuration with each optimizer
for config_name, config in configs.items():
    for opt_name in optimizer_configs.keys():
        cv_scores = []
        
        print(f"\nTraining {config['name']} with {opt_name} optimizer")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val_processed)):
            X_train_fold = X_train_val_processed[train_idx]
            y_train_fold = y_train_val.iloc[train_idx]
            X_val_fold = X_train_val_processed[val_idx]
            y_val_fold = y_train_val.iloc[val_idx]
            
            # Create and train model with a fresh optimizer instance
            model = create_model(config, opt_name, input_shape)
            
            # Train the model
            history = model.fit(
                X_train_fold, y_train_fold,
                epochs=100,
                batch_size=16,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate on validation fold
            val_mse = model.evaluate(X_val_fold, y_val_fold, verbose=0)[0]
            y_val_pred = model.predict(X_val_fold, verbose=0)
            val_r2 = r2_score(y_val_fold, y_val_pred.flatten())
            
            cv_scores.append({
                'mse': val_mse,
                'r2': val_r2
            })
            
            print(f"  Fold {fold+1}: MSE = {val_mse:.4f}, R² = {val_r2:.4f}")
        
        # Calculate average scores across folds
        avg_mse = np.mean([score['mse'] for score in cv_scores])
        avg_r2 = np.mean([score['r2'] for score in cv_scores])
        
        results.append({
            'config': config_name,
            'config_name': config['name'],
            'optimizer': opt_name,
            'avg_mse': avg_mse,
            'avg_r2': avg_r2
        })
        
        print(f"  Average: MSE = {avg_mse:.4f}, R² = {avg_r2:.4f}")

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print("\nCross-validation results:")
print(results_df)

# Find the best model based on MSE
best_model_mse = results_df.loc[results_df['avg_mse'].idxmin()]
print("\nBest model based on MSE:")
print(best_model_mse)

# Find the best model based on R²
best_model_r2 = results_df.loc[results_df['avg_r2'].idxmax()]
print("\nBest model based on R²:")
print(best_model_r2)

# Train the best model on the entire training+validation set
best_config = configs[best_model_r2['config']]
best_opt_name = best_model_r2['optimizer']

print(f"\nTraining the best model ({best_config['name']} with {best_opt_name} optimizer) on the entire training set...")

# Create and train the best model
best_model = create_model(best_config, best_opt_name, input_shape)
best_model.fit(
    X_train_val_processed, y_train_val,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate on test set
y_pred = best_model.predict(X_test_processed, verbose=0).flatten()
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("\nTest set evaluation:")
print(f"MSE: {test_mse:.4f}")
print(f"R²: {test_r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs Predicted MPG')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.close()

# Plot MSE for each configuration and optimizer
plt.figure(figsize=(12, 6))
bar_width = 0.25
index = np.arange(len(configs))

for i, opt in enumerate(optimizer_configs.keys()):
    opt_results = results_df[results_df['optimizer'] == opt]
    mse_values = []
    
    for config in configs.keys():
        config_opt_result = opt_results[opt_results['config'] == config]
        mse_values.append(config_opt_result['avg_mse'].values[0])
    
    plt.bar(index + i * bar_width, mse_values, bar_width, label=opt)

plt.xlabel('Model Configuration')
plt.ylabel('Average MSE')
plt.title('MSE by Model Configuration and Optimizer')
plt.xticks(index + bar_width, configs.keys())
plt.legend()
plt.grid(True, axis='y')
plt.savefig('mse_comparison.png')
plt.close()

# Plot R² for each configuration and optimizer
plt.figure(figsize=(12, 6))

for i, opt in enumerate(optimizer_configs.keys()):
    opt_results = results_df[results_df['optimizer'] == opt]
    r2_values = []
    
    for config in configs.keys():
        config_opt_result = opt_results[opt_results['config'] == config]
        r2_values.append(config_opt_result['avg_r2'].values[0])
    
    plt.bar(index + i * bar_width, r2_values, bar_width, label=opt)

plt.xlabel('Model Configuration')
plt.ylabel('Average R²')
plt.title('R² by Model Configuration and Optimizer')
plt.xticks(index + bar_width, configs.keys())
plt.legend()
plt.grid(True, axis='y')
plt.savefig('r2_comparison.png')
plt.close()

# Summary of all configurations
print("\nSummary of all configurations:")
print("-----------------------------")
for config_name, config in configs.items():
    print(f"Configuration {config_name}: {config['name']}")
    print(f"  Layers: {config['layers']}")
    print("  Performance with different optimizers:")
    
    for opt_name in optimizer_configs.keys():
        config_opt_result = results_df[(results_df['config'] == config_name) & (results_df['optimizer'] == opt_name)]
        print(f"    {opt_name}: MSE = {config_opt_result['avg_mse'].values[0]:.4f}, R² = {config_opt_result['avg_r2'].values[0]:.4f}")
    
    print()

print("\nConclusion:")
print(f"The best model is {best_model_r2['config_name']} with {best_model_r2['optimizer']} optimizer")
print(f"Cross-validation R²: {best_model_r2['avg_r2']:.4f}")
print(f"Test set R²: {test_r2:.4f}")
print(f"Test set MSE: {test_mse:.4f}")