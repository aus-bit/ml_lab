import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = fetch_openml("boston", version=1, as_frame=True)
X = data.data[["RM"]].astype(float)  # average number of rooms
y = data.target.astype(float)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
train_errors, val_errors = [], []
degrees = range(1, 10)
for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    val_errors.append(mean_squared_error(y_val, model.predict(X_val)))
plt.plot(list(degrees), train_errors, marker="o", label="Training error")
plt.plot(list(degrees), val_errors, marker="o", label="Validation error")
plt.xlabel("Polynomial degree"); plt.ylabel("MSE"); plt.title("Bias-Variance Tradeoff")
plt.legend(); plt.show()
print("Training errors:", train_errors)
print("Validation errors:", val_errors)
