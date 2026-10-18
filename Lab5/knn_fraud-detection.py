#NAME HUSNAIN NADEEM 
# KNN for Credit Card Fraud Detection

import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Step 1: Download dataset from Kaggle
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)

# Step 2: Load the dataset
data = pd.read_csv(os.path.join(path, "creditcard.csv"))

# Step 3: Display first few rows
print("\nDataset Preview:")
print(data.head())

# Step 4: Display dataset information
print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

# Step 5: Check for missing values
print("\nMissing Values:")
print(data.isnull().sum().sum())

# Step 6: Display target distribution
print("\nTarget Distribution:")
print(data["Class"].value_counts())

print("\nFraud Percentage:")
print(f"{data['Class'].mean() * 100:.3f}%")

# Step 7: Visualize target distribution
plt.figure(figsize=(8, 5))
data["Class"].value_counts().sort_index().plot(kind="bar")
plt.title("Target Distribution")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 8: Split dataset into features X and target y
X = data.drop("Class", axis=1)
y = data["Class"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Step 9: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 10: Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed")

# Step 11: Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nClass Distribution Before SMOTE:")
print(y_train.value_counts())

print("\nClass Distribution After SMOTE:")
print(y_train_smote.value_counts())

# Step 12: Train KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_smote, y_train_smote)

# Predict on testing data
y_pred = knn.predict(X_test_scaled)

# Step 13: Evaluate baseline KNN model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nKNN Model Results (k=5)")
print("Accuracy:", round(accuracy, 4))
print("F1-Score:", round(f1, 4))

# Step 14: Tune k using GridSearchCV
# Scaling and SMOTE are performed inside each CV fold to prevent data leakage.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier(n_jobs=-1))
])

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest k Value:")
print(grid_search.best_params_["knn__n_neighbors"])

print("\nBest Cross-Validation F1-Score:")
print(round(grid_search.best_score_, 4))

# Step 15: Predict using the tuned model
best_knn = grid_search.best_estimator_
y_pred_tuned = best_knn.predict(X_test)

# Step 16: Evaluate tuned KNN model
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)

print("\nTuned KNN Model Results")
print("Accuracy:", round(accuracy_tuned, 4))
print("F1-Score:", round(f1_tuned, 4))

# Step 17: Compare both models
results = pd.DataFrame({
    "Model": ["KNN (k=5)", "Tuned KNN"],
    "Accuracy": [accuracy, accuracy_tuned],
    "F1-Score": [f1, f1_tuned]
})

print("\nModel Performance Comparison:")
print(results.to_string(index=False))

# Step 18: Display classification report
print("\nClassification Report (Tuned KNN):")
print(classification_report(
    y_test,
    y_pred_tuned,
    target_names=["Normal", "Fraud"],
    zero_division=0
))

# Step 19: Display confusion matrix
cm = confusion_matrix(y_test, y_pred_tuned)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Tuned KNN")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks([0, 1], ["Normal", "Fraud"])
plt.yticks([0, 1], ["Normal", "Fraud"])
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.show()

# Step 20: Plot model performance comparison
results.set_index("Model")[["Accuracy", "F1-Score"]].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("KNN Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
