import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("Car Price Prediction.csv")

X = df[["horsepower"]].values
y = df["price"].values

plt.scatter(X, y)

x_grid = np.arange(X.min(), X.max(), 1).reshape(-1, 1)

for degree in [1, 2, 3, 4]:

    poly = PolynomialFeatures(degree=degree)

    X_poly = poly.fit_transform(X)
    x_grid_poly = poly.transform(x_grid)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_grid = model.predict(x_grid_poly)

    plt.plot(x_grid, y_grid, label=f"Degree {degree}")

plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Polynomial Regression Comparison")
plt.legend()
plt.show()