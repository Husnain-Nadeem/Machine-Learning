#Husnain Nadeem -1038

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from xgboost import XGBClassifier


# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 10 Rows:")
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())


# Convert TotalCharges to numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nMissing TotalCharges:", df["TotalCharges"].isnull().sum())

# Fill missing values with the median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)


# Drop customerID and encode Churn
df.drop("customerID", axis=1, inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]


# Convert categorical columns into numeric columns
X = pd.get_dummies(X, drop_first=True, dtype=int)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# Train the default boosting models
gb_model = GradientBoostingClassifier(random_state=42)
ada_model = AdaBoostClassifier(random_state=42)
xgb_model = XGBClassifier(random_state=42)


models = {
    "Gradient Boosting": gb_model,
    "AdaBoost": ada_model,
    "XGBoost Default": xgb_model
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "trained successfully.")


# Tune XGBoost using GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-Score:")
print(round(grid_search.best_score_, 4))


# Evaluate default and tuned XGBoost
tuned_xgb = grid_search.best_estimator_

comparison_models = {
    "XGBoost Default": xgb_model,
    "XGBoost Tuned": tuned_xgb
}

results = []

for name, model in comparison_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })


# Display the results
results_df = pd.DataFrame(results)

print("\nDefault vs Tuned XGBoost:")
print(results_df.to_string(index=False, formatters={
    "Accuracy": "{:.4f}".format,
    "Precision": "{:.4f}".format,
    "Recall": "{:.4f}".format,
    "F1-Score": "{:.4f}".format,
    "ROC-AUC": "{:.4f}".format
}))


# Plot the five most important features
importance = pd.Series(
    tuned_xgb.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(5)

print("\nTop 5 Important Features:")
print(importance)

importance.sort_values().plot(
    kind="barh",
    figsize=(8, 5)
)

plt.title("Top 5 Important Features - Tuned XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
