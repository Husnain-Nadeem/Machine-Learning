from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('income_evaluation.csv')
print(df.head())

categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(exclude=['object']).columns

df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

X = df.drop("income_ >50K", axis=1)
y = df["income_ >50K"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(
    X_train[numerical_features]
)

X_test[numerical_features] = scaler.transform(
    X_test[numerical_features]
)

print(X_train.head())

svc = SVC(random_state=101)
svc.fit(X_train, y_train)

accuracies = cross_val_score(svc, X_train, y_train, cv=5)

print("Train Score:", np.mean(accuracies))
print("Test Score:", svc.score(X_test, y_test))

grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ["linear", "poly", "rbf", "sigmoid"],
    'degree': [1, 3, 5, 7],
    'gamma': [0.01, 1]
}

svm = SVC()

svm_cv = GridSearchCV(svm, grid, cv=2)
svm_cv.fit(X_train, y_train)

print("Best Parameters:", svm_cv.best_params_)
print("Train Score:", svm_cv.best_score_)
print("Test Score:", svm_cv.score(X_test, y_test))

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rnd_clf),
        ('svc', svm_clf)
    ],
    voting='hard'
)

voting_clf.fit(X_train, y_train)

print(voting_clf.predict(X_test.iloc[[0]]))

print("Number of features:", X_train.shape[1])
print("Training data shape:", X_train.shape)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

mushroom_data = pd.read_csv("mushroomss.csv")

print("Dataset Preview:")
print(mushroom_data.head())

print("\nMissing Values in Each Column:")
print(mushroom_data.isnull().sum())

print(mushroom_data)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.countplot(data=mushroom_data, x='class')
plt.title('Target Distribution')

plt.subplot(2, 3, 2)
sns.countplot(data=mushroom_data, x='cap-shape', hue='class')
plt.title('Cap Shape vs Class')

plt.subplot(2, 3, 3)
sns.countplot(data=mushroom_data, x='cap-color', hue='class')
plt.title('Cap Color vs Class')

plt.show()

print(mushroom_data.columns)