
import pandas as pd


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# dataset load kar rahe hain
df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print(df["Class"].value_counts())

# features aur target alag kar rahe hain
X = df.drop("Class", axis=1)
y = df["Class"]

# data ko training aur testing mein divide kar rahe hain
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# scaling, SMOTE aur KNN ko aik pipeline mein rakh rahe hain
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# k=5 wala model train kar rahe hain
pipeline.fit(X_train, y_train)

# test data par prediction kar rahe hain
y_pred = pipeline.predict(X_test)

# model ki performance check kar rahe hain
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nKNN Results (k=5)")
print("Accuracy:", round(accuracy, 4))
print("F1-Score:", round(f1, 4))

# different k values check kar rahe hain
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

# best k find karne ke liye model tune kar rahe hain
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best CV F1-Score:", round(grid_search.best_score_, 4))

# best model se prediction kar rahe hain
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

# tuned model ki performance check kar rahe hain
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_f1 = f1_score(y_test, y_pred_tuned, zero_division=0)

print("\nTuned KNN Results")
print("Accuracy:", round(tuned_accuracy, 4))
print("F1-Score:", round(tuned_f1, 4))

# dono models ke results compare kar rahe hain
results = pd.DataFrame({
    "Model": ["KNN (k=5)", "Tuned KNN"],
    "Accuracy": [accuracy, tuned_accuracy],
    "F1-Score": [f1, tuned_f1]
})

print("\nModel Comparison")
print(results.to_string(index=False))
