import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score

# Generate dummy data
m = 1000
X = np.linspace(-1, 1, num=m).reshape(m, 1)
Y = 4 * X * np.sin(10 * X) + X * np.exp(X) * np.cos(20 * X)

# Scale the data
X_scaled = scale(X)
Y_scaled = scale(Y)

# Plot original data
plt.figure()
plt.plot(X_scaled, Y_scaled, linewidth=3)
plt.grid()
plt.title("Original Data")
plt.show()

# Create and train the MLP model
model = MLPRegressor(hidden_layer_sizes=(40, 30, 20, 10),
                     activation='tanh',
                     solver='sgd',
                     learning_rate_init=0.001,
                     max_iter=50000,
                     batch_size=100,
                     verbose=True)

model.fit(X_scaled, Y_scaled.ravel())

# Predict
Yhat_scaled = model.predict(X_scaled)

# Evaluate
r2 = r2_score(Y_scaled, Yhat_scaled)
print(f"R² Score: {r2:.4f}")

# Plot predictions
plt.figure(figsize=(15, 8))
plt.plot(X_scaled[:, 0], Y_scaled, label='Y real', linewidth=3)
plt.plot(X_scaled[:, 0], Yhat_scaled, label='Y estimated', linewidth=3)
plt.title('Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()
