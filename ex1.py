import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X = data.data[:, [data.feature_names.index("AveRooms")]]
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normal equation: theta = inverse(X'X) X'y
X_b = np.c_[np.ones((len(X_train), 1)), X_train]
theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_train
pred_normal = np.c_[np.ones((len(X_test), 1)), X_test] @ theta

# Gradient descent
X_mean, X_std = X_train.mean(), X_train.std()
Xs = (X_train - X_mean) / X_std
X_test_s = (X_test - X_mean) / X_std
X_g = np.c_[np.ones((len(Xs), 1)), Xs]
w = np.zeros(2)
lr = 0.01
for _ in range(2000):
    error = X_g @ w - y_train
    w -= lr * (2 / len(X_g)) * X_g.T @ error
pred_gd = w[0] + w[1] * ((X_test.ravel() - X_mean) / X_std)

print("Normal equation MSE:", mean_squared_error(y_test, pred_normal))
print("Normal equation R2:", r2_score(y_test, pred_normal))
print("Gradient descent MSE:", mean_squared_error(y_test, pred_gd))
print("Gradient descent R2:", r2_score(y_test, pred_gd))

model = LinearRegression().fit(X_train, y_train)
order = np.argsort(X_test.ravel())
plt.scatter(X_test, y_test, s=10, alpha=0.4)
plt.plot(X_test.ravel()[order], model.predict(X_test)[order], color="red")
plt.xlabel("Average rooms"); plt.ylabel("House value"); plt.title("Linear Regression")
plt.show()
