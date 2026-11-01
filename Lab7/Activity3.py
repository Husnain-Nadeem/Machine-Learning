#Husnain Nadeem -1038
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv("insurance.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 10 Rows:")
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCharges Summary:")
print(df["charges"].describe())


# Encode categorical columns
df["sex"] = df["sex"].map({"female": 0, "male": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})

df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)


# Separate features and target
X = df.drop("charges", axis=1)
y = df["charges"]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


# Create the base models
ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

rf = RandomForestRegressor(random_state=42)

gb = GradientBoostingRegressor(random_state=42)


# Create the stacking model
stack = StackingRegressor(
    estimators=[
        ("ridge", ridge),
        ("rf", rf),
        ("gb", gb)
    ],
    final_estimator=LinearRegression(),
    cv=5
)


# Train all four models
models = {
    "Ridge": ridge,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Stacking": stack
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred)
    })


# Compare the models
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=False, formatters={
    "MAE": "{:.2f}".format,
    "RMSE": "{:.2f}".format,
    "R2": "{:.4f}".format
}))


# Print the final estimator coefficients
print("\nStacking Model Coefficients:")

for name, coef in zip(
    stack.named_estimators_.keys(),
    stack.final_estimator_.coef_
):
    print(f"{name}: {coef:.4f}")

print("Intercept:", round(stack.final_estimator_.intercept_, 4))


# Plot actual versus predicted charges
stack_predictions = stack.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, stack_predictions, alpha=0.7)

min_value = min(y_test.min(), stack_predictions.min())
max_value = max(y_test.max(), stack_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges - Stacking")
plt.tight_layout()
plt.show()
