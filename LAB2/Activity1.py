#Name Husnain Nadeem
#reg no 23-ntu-cs-1038
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load data
df = pd.read_csv("Medical Cost Personal Datasets.csv")

# quick look at dataset
print(df.head())
print(df.info())
print(df.describe())

# features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# columns
num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

# preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols)
])

# model pipeline
model = Pipeline([
    ("prep", preprocessor),
    ("lr", LinearRegression())
])

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# train model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Predicted vs Actual Costs")
plt.show()