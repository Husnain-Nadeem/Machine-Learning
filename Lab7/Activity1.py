import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Load the dataset
df = pd.read_csv("water_potability.csv")

print("Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))

print("\nMissing values:")
print(df.isnull().sum())

print("\nPotability distribution:")
print(df["Potability"].value_counts())


# Separate features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# Logistic Regression
lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


# SVC
svc = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", SVC(probability=True, random_state=42))
])


# Random Forest
rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(random_state=42))
])


# Hard and soft voting
hard_voting = VotingClassifier(
    estimators=[("lr", lr), ("svc", svc), ("rf", rf)],
    voting="hard"
)

soft_voting = VotingClassifier(
    estimators=[("lr", lr), ("svc", svc), ("rf", rf)],
    voting="soft"
)


# Bagging with 10 trees
bagging_10 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=10,
        max_samples=0.5,
        bootstrap=True,
        random_state=42
    ))
])


# Bagging with 100 trees
bagging_100 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        max_samples=0.5,
        bootstrap=True,
        random_state=42
    ))
])


# Pasting with 100 trees
pasting_100 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        max_samples=0.5,
        bootstrap=False,
        random_state=42
    ))
])


# Single decision tree
dt = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DecisionTreeClassifier(random_state=42))
])


# Train and test all models
models = {
    "Logistic Regression": lr,
    "SVC": svc,
    "Random Forest": rf,
    "Hard Voting": hard_voting,
    "Soft Voting": soft_voting,
    "Decision Tree": dt,
    "Bagging 10": bagging_10,
    "Bagging 100": bagging_100,
    "Pasting 100": pasting_100
}

results = []
matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    })

    matrices[name] = confusion_matrix(y_test, y_pred)


# Compare model results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("F1-Score", ascending=False)

print("\nModel Comparison:")
print(results_df.to_string(index=False, formatters={
    "Accuracy": "{:.4f}".format,
    "Precision": "{:.4f}".format,
    "Recall": "{:.4f}".format,
    "F1-Score": "{:.4f}".format
}))


# Show the best model's confusion matrix
best_model = results_df.iloc[0]["Model"]

print("\nBest Model:", best_model)

sns.heatmap(matrices[best_model], annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - " + best_model)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Compare the best single model with the best ensemble
single_models = ["Logistic Regression", "SVC", "Random Forest", "Decision Tree"]

ensemble_models = [
    "Hard Voting", "Soft Voting",
    "Bagging 10", "Bagging 100", "Pasting 100"
]

single_results = results_df[results_df["Model"].isin(single_models)]
ensemble_results = results_df[results_df["Model"].isin(ensemble_models)]

best_single = single_results.iloc[0]
best_ensemble = ensemble_results.iloc[0]

print("\nBest Single Model:", best_single["Model"])
print("F1-Score:", round(best_single["F1-Score"], 4))

print("\nBest Ensemble Model:", best_ensemble["Model"])
print("F1-Score:", round(best_ensemble["F1-Score"], 4))

if best_ensemble["F1-Score"] > best_single["F1-Score"]:
    print("\nThe best ensemble performed better than the best single model.")
else:
    print("\nThe best ensemble did not perform better than the best single model.")
