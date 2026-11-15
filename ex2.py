
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Step 1: Load Auto MPG dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"

df = pd.read_csv(url)

# Step 2: Clean the dataset
df = df.replace("?", np.nan)
df["horsepower"] = pd.to_numeric(
    df["horsepower"], errors="coerce"
)
df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce")
df = df.dropna()

# Step 3: Select input and output
X = df[["horsepower"]]
y = df["mpg"]

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Compare different polynomial degrees
degrees = [1, 2, 3, 4, 5]

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        StandardScaler(),
        LinearRegression()
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict output
    y_pred = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nPolynomial Degree:", degree)
    print("Mean Squared Error:", round(mse, 3))
    print("R2 Score:", round(r2, 3))

# Step 6: Plot polynomial regression curves
plt.figure(figsize=(9, 6))

X_plot = pd.DataFrame({
    "horsepower": np.linspace(
        X["horsepower"].min(),
        X["horsepower"].max(),
        200
    )
})

plt.scatter(X, y, color="blue", label="Actual Data")

for degree in [1, 2, 3, 5]:

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        StandardScaler(),
        LinearRegression()
    )

    model.fit(X, y)

    y_plot = model.predict(X_plot)

    plt.plot(
        X_plot["horsepower"],
        y_plot,
        label=f"Degree {degree}"
    )

plt.xlabel("Horsepower")
plt.ylabel("Miles per Gallon (MPG)")
plt.title("Polynomial Regression - Auto MPG Dataset")
plt.legend()
plt.grid(True)
plt.show()

