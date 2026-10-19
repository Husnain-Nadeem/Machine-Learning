# NAME: Husnain Nadeem
# STUDENT ID: 23-ntu-cs-1038

# PCA is used to reduce the dimensionality of data while retaining most of its variance.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine, load_digits

wine = load_wine()

X = wine.data
y = wine.target
feat_name = wine.feature_names
target_names = wine.target_names

print("Original data shape:", X.shape)
print("Original target shape:", y.shape)
print("Feature names:", feat_name)
print("Target names:", target_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print("Original dataset shape:", X.shape)
print("Reduced dataset shape:", X_reduced.shape)

wine_colors = ['orangered', 'darkred', 'crimson']

plt.figure(figsize=(10, 6))

for class_num, color, target_name in zip(
    [0, 1, 2], wine_colors, target_names
):
    plt.scatter(
        X_reduced[y == class_num, 0],
        X_reduced[y == class_num, 1],
        color=color,
        label=target_name,
        alpha=0.7
    )

plt.title("PCA of Wine Dataset")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

print("Principal component loadings:")
print(pca.components_)
print("Shape of principal components:", pca.components_.shape)

plt.figure(figsize=(14, 5))

sns.heatmap(
    pca.components_,
    cmap='viridis',
    xticklabels=feat_name,
    yticklabels=['First Component', 'Second Component']
)

plt.title("PCA Component Loadings")
plt.xlabel("Original Features")
plt.ylabel("Principal Components")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("Explained variance:")
print(pca.explained_variance_)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

pca_96 = PCA(n_components=0.96)
X_reduced_96 = pca_96.fit_transform(X_scaled)

print("Number of components selected:", pca_96.n_components_)
print(
    "Total explained variance:",
    np.sum(pca_96.explained_variance_ratio_)
)

print("Reduced dataset shape:", X_reduced_96.shape)

print(
    "Variance explained by first component:",
    round(pca.explained_variance_ratio_[0] * 100, 2), "%"
)

print(
    "Variance explained by second component:",
    round(pca.explained_variance_ratio_[1] * 100, 2), "%"
)

print(
    "Total variance explained by two components:",
    round(np.sum(pca.explained_variance_ratio_) * 100, 2), "%"
)

pca_all = PCA()
pca_all.fit(X_scaled)

exp_var_ratio = pca_all.explained_variance_ratio_
cumsum = np.cumsum(exp_var_ratio)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumsum) + 1),
    cumsum,
    marker='o'
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.xticks(range(1, len(cumsum) + 1))
plt.grid()
plt.tight_layout()
plt.show()

digits = load_digits()

digits_data = digits.data
digits_target = digits.target

print("Original Digits dataset shape:", digits_data.shape)

digits_pca = PCA(n_components=2)
digits_reduced = digits_pca.fit_transform(digits_data)

print("Reduced Digits dataset shape:", digits_reduced.shape)

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    digits_reduced[:, 0],
    digits_reduced[:, 1],
    c=digits_target,
    cmap='jet',
    edgecolors='none'
)

plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA Visualization of Digits Dataset")
plt.colorbar(scatter, label="Digit Class")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()