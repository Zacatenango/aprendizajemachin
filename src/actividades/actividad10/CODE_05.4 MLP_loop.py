
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
df = sns.load_dataset('mpg')

df.isna().sum()

df.dropna(inplace=True)

df.drop('name', axis=1, inplace=True)

df['origin'].replace({'usa': 1, 'europe': 2, 'japan': 3}, inplace=True)

df['origin'] = df['origin'].astype('category').cat.codes

y = df['mpg']
X = df.drop('mpg', axis=1)
#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_dev, X_final, y_dev, y_final = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.3, random_state=42)

architectures = [
    (4, 4, 4),
    (12,),
    (3, 3, 3, 3)
]

optimizers = ['adam', 'sgd']
activation = ['relu', 'tanh', 'identity']

results = []

for arch in architectures:
    for solver in optimizers:
        for act_func in activation:
            mlp = MLPRegressor(
                hidden_layer_sizes=arch,
                activation=act_func,
                solver=solver,
                max_iter=20000,
                random_state=42,
                early_stopping=True
            )

        scores = cross_val_score(mlp, X_dev, y_dev, cv=3, scoring='r2')
        avg_r2 = np.mean(scores)
        std_r2 = np.std(scores)
        results.append({
            'architecture': arch,
            'optimizer': solver,
            'act':act_func,
            'avg_r2': avg_r2,
            'std_r2': std_r2
        })
        print(f"Arch: {arch}, Optimizer: {solver}, R²: {avg_r2:.3f} ± {std_r2:.3f}")

best_result = max(results, key=lambda x: x['avg_r2'])
print("\nBest combination found:")
print(f"Architecture: {best_result['architecture']}")
print(f"activation: {best_result['act']}")
print(f"Optimizer: {best_result['optimizer']}")
print(f"Average R²: {best_result['avg_r2']:.3f}")

#%%
final_mlp = MLPRegressor(
    hidden_layer_sizes=best_result['architecture'],
    activation='relu',
    solver=best_result['optimizer'],
    max_iter=2000,
    random_state=42
)

final_mlp.fit(X_train, y_train)

y_pred = final_mlp.predict(X_test)
final_r2 = r2_score(y_test, y_pred)
final_mse = mean_squared_error(y_test, y_pred)

print("\nFinal model performance on test set:")
print(f"R² Score: {final_r2:.3f}")
print(f"Mean Squared Error (MSE): {final_mse:.3f}")
#%%
y_final_pred = final_mlp.predict(X_final)
final_holdout_r2 = r2_score(y_final, y_final_pred)
final_holdout_mse = mean_squared_error(y_final, y_final_pred)

print("\nPerformance on final holdout set:")
print(f"R² Score: {final_holdout_r2:.3f}")
print(f"Mean Squared Error (MSE): {final_holdout_mse:.3f}")
