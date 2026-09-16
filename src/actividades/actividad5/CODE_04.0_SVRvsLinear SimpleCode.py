import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Data
X = np.array([9.00, 1.00, 7.00, 9.00, 18.00, 18.00, 21.00, 4.00, 9.00, 7.00]).reshape(-1, 1)
Y = np.array([204.00, 295.00, 257.00, 237.00, 175.00, 164.00, 150.00, 277.00, 235.00, 258.00])

# Linear Regression model
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

# Getting the slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Plotting X vs Y with regression line
plt.scatter(X, Y, color='blue', label="Data points")
plt.plot(X, Y_pred, color='red', label=f"Linear fit: Y = {slope:.2f}X + {intercept:.2f}")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y with Linear Regression')
plt.legend()
plt.show()

# Linear equation
linear_equation = f"Y = {slope:.2f}X + {intercept:.2f}"
print(f"Linear equation: {linear_equation}")

# Measuring Mean Squared Error (MSE)
mse = mean_squared_error(Y, Y_pred)
print(f"Mean Squared Error: {mse:.2f}")

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error





# Data
# Podemos ver aquí que SVR es muy robusto a outliers
Y = np.array([204.00, 295.00, 257.00, 237.00, 175.00, 164.00, 150.00, 277.00, 235.00, 258.00])
Y = np.array([204.00, 195.00, 257.00, 237.00, 175.00, 164.00, 150.00, 277.00, 235.00, 258.00])
Y = np.array([204.00, 195.00, 257.00, 237.00, 175.00, 264.00, 150.00, 277.00, 235.00, 258.00])
Y = np.array([204.00, 95.00, 257.00, 237.00, 175.00, 264.00, 190, 277.00, 235.00, 258.00])


# Linear Regression model
lin_model = LinearRegression()
lin_model.fit(X, Y)
Y_pred_lin = lin_model.predict(X)

# SVR model
svr_model = SVR(kernel='linear',C=1, epsilon=1)
svr_model.fit(X, Y)
Y_pred_svr = svr_model.predict(X)

#High C Lower error on training set, but higher risk of overfitting
#Minimize the error on the training data, fitting even noise.
#Low  C Higher error on training set, but better generalization.
#more deviations from the exact targets

#epsilon: Margin of tolerance where no penalty

# Create subplots


# Plot Linear Regression
plt.scatter(X, Y, color='blue', label="Data points")
plt.plot(X, Y_pred_lin, color='red', label='Linear regression')
plt.plot(X, Y_pred_svr, color='green', label='SVR')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression: X vs Y')

plt.tight_layout()
plt.show()

R2Lin = lin_model.score(X,Y_pred_svr)

R2SVR = svr_model.score(X,Y_pred_lin)


# Measuring Mean Squared Errors (MSE)
mse_lin = mean_squared_error(Y, Y_pred_lin)
mse_svr = mean_squared_error(Y, Y_pred_svr)

print(f"Linear Regression MSE: {mse_lin:.2f}")
print(f"Linear Regression R2: {R2Lin:.2f}")


print(f"SVR MSE: {mse_svr:.2f}")
print(f"SVR R2: {R2SVR:.2f}")


#%%

