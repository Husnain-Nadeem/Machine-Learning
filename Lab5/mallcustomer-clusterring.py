import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# Step 1: Load and display the dataset
data = pd.read_csv("Mall_Customers.csv")

print("First five rows:")
print(data.head())

print("\nDataset information:")
data.info()

print("\nMissing values:")
print(data.isnull().sum())

# Step 2: Select Annual Income and Spending Score
features = data[["Annual Income (k$)", "Spending Score (1-100)"]]
features = features.fillna(features.mean())

# Step 3: Apply K-Means without scaling
k = 5

kmeans_original = KMeans(
    n_clusters=k, random_state=42, n_init=10
)
labels_original = kmeans_original.fit_predict(features)

plt.figure(figsize=(7, 5))
plt.scatter(
    features["Annual Income (k$)"],
    features["Spending Score (1-100)"],
    c=labels_original, cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Without Scaling")
plt.show()

# Step 4: Scale features for DBSCAN
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 5: Apply DBSCAN
dbscan = DBSCAN(eps=0.35, min_samples=5)
labels_dbscan = dbscan.fit_predict(features_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(
    features["Annual Income (k$)"],
    features["Spending Score (1-100)"],
    c=labels_dbscan, cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clusters (-1 = Noise)")
plt.show()

print("\nDBSCAN cluster counts:")
print(pd.Series(labels_dbscan).value_counts().sort_index())

# Step 6: Apply K-Means after scaling
kmeans_scaled = KMeans(
    n_clusters=k, random_state=42, n_init=10
)
labels_scaled = kmeans_scaled.fit_predict(features_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(
    features["Annual Income (k$)"],
    features["Spending Score (1-100)"],
    c=labels_scaled, cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means After StandardScaler")
plt.show()

# Step 7: Compare clusters before and after scaling
ari = adjusted_rand_score(labels_original, labels_scaled)

print("\nAdjusted Rand Index:", round(ari, 3))

if ari == 1:
    print("Scaling did not change the cluster assignments.")
else:
    print("Scaling changed some cluster assignments.")

# Step 8: Elbow Method
inertias = []

for i in range(1, 11):
    model = KMeans(
        n_clusters=i, random_state=42, n_init=10
    )
    model.fit(features_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.show()

# Step 9: Interpret clusters
summary = features.copy()
summary["Cluster"] = labels_scaled

print("\nCluster averages:")
print(summary.groupby("Cluster").mean().round(2))

income_median = features["Annual Income (k$)"].median()
spending_median = features["Spending Score (1-100)"].median()

print("\nCluster Interpretation:")

for cluster, row in summary.groupby("Cluster").mean().iterrows():
    income = row["Annual Income (k$)"]
    spending = row["Spending Score (1-100)"]

    if income >= income_median and spending >= spending_median:
        description = "High income, high spending"
    elif income >= income_median and spending < spending_median:
        description = "High income, low spending"
    elif income < income_median and spending >= spending_median:
        description = "Low income, high spending"
    else:
        description = "Low income, low spending"

    print("Cluster", cluster, ":", description)

# Step 10: Add a random LuckyNumber feature
np.random.seed(42)
data["LuckyNumber"] = np.random.randint(0, 101, size=len(data))

print("\nDataset with LuckyNumber:")
print(data.head())

# Scale all three features
features_lucky = data[
    ["Annual Income (k$)", "Spending Score (1-100)", "LuckyNumber"]
].copy()

features_lucky = features_lucky.fillna(features_lucky.mean())

scaler_lucky = StandardScaler()
features_lucky_scaled = scaler_lucky.fit_transform(features_lucky)

# Run K-Means with LuckyNumber
kmeans_lucky = KMeans(
    n_clusters=k, random_state=42, n_init=10
)
labels_lucky = kmeans_lucky.fit_predict(features_lucky_scaled)

# Compare assignments
ari_lucky = adjusted_rand_score(labels_scaled, labels_lucky)

print("\nComparison after adding LuckyNumber:")
print("Adjusted Rand Index:", round(ari_lucky, 3))

# Plot clusters with LuckyNumber included
plt.figure(figsize=(7, 5))
plt.scatter(
    features["Annual Income (k$)"],
    features["Spending Score (1-100)"],
    c=labels_lucky, cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means After Adding LuckyNumber")
plt.show()

print("\nConclusion:")
print("Scaling changes the relative influence of features on distance.")
print("The Elbow plot helps estimate the best value of k.")
print("LuckyNumber is random and has no real customer meaning.")
print("It can change clusters because K-Means uses distances between features.")
print("Avoid irrelevant features when building meaningful customer segments.")