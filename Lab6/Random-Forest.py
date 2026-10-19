import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/cpu-performance/machine.data"
)

columns = [
    "vendor", "model", "MYCT", "MMIN", "MMAX",
    "CACH", "CHMIN", "CHMAX", "PRP", "ERP"
]

data = pd.read_csv(url, names=columns)

print("First five rows:")
print(data.head())

print("\nDataset shape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())

data.hist(figsize=(12, 8), bins=15)
plt.tight_layout()
plt.show()

correlation = data.select_dtypes(include=np.number).corr()

print("\nCorrelation with PRP:")
print(correlation["PRP"].sort_values(ascending=False))

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

X = data.drop(columns=["PRP", "ERP", "vendor", "model"])
y = data["PRP"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

forest_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

forest_reg.fit(X_train, y_train)

y_train_pred = forest_reg.predict(X_train)

train_rmse = np.sqrt(
    mean_squared_error(y_train, y_train_pred)
)

train_r2 = r2_score(y_train, y_train_pred)

print("\nInitial Training Results")
print("Training RMSE:", round(train_rmse, 2))
print("Training R2 Score:", round(train_r2, 4))