#NAME : Husnain Nadeem
# STUDENT ID : 23-ntu-cs-1038
#PCA unspervised learning technique is used to reduce the dimensionality of the data while retaining most of the variance in the data. It does this by finding new axes (principal components) that maximize the variance in the data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine , load_digits

wine = load_wine()
#data load_wine()
#data and target
X= wine.data
y= wine.target
feat_name = wine.feature_names
target_names = wine.target_names

print ("orignal data shape : ", X.shape)
print ("orignal target shape : ", y.shape)
print ("Feature Names : ", feat_name)
print ("Target Names : ", target_names)


#step1 : Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#step2 : reduce the data tp two principal cpomponent
pca = PCA(n_components=2)
X_reduced =pca.fit_transform(X_scaled)

print("orignal dataset shape:" , X.shape)
print("Redced dataset shape:", X_reduced.shape)

wine_colors= ['orangered' , 'darkred' , 'crimson']

plt.figure(figsize=(10,6))
for class_num , color, targer_name in zip (
    [0,1,2], wine_colors, target_names
):
        plt.scatter(
                X_reduced[y==class_num,0],
                X_reduced[y==class_num,1],
                color=color,
                label=target_names,
                alpha=0.7
        )
plt.title('PCA OF WINE DATASET')
plt.xlabel('First princial component')
plt.ylabel('Second principal component')
plt.legend()
plt.grid(alpha=0.2)
plt.show()

print("Principal component loading:")
print(pca.components_)
print("shape of principal component ", pca.components_.shape)

plt.figure(figsize=(14,5))
sns.heatmap(
        pca.components_,
        cmap='viridis',
        xticklabels=feat_name,
        yticklabels=['First component ', 'second component']

)

plt.title('PCA Component Loading')
plt.xlabel('Orignam Features')
plt.ylabel("principal components")
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
plt.show()

print("Explained variance:")
print(pca.explained_variance_)

print ( "\nExplained variance ratio ")
print (pca.explained_variance_ratio_)


pca_3 =PCA(n_components=0.96)
X_reduced_96 = pca_3.fit_transform(X_scaled)

print("Number of components selected :" , pca_3.n_components_)
print("Total explained variance :", 
      np.sim(pca_3.explained_variance_ratio_))

print("Reduced dataset shape: ", X_reduced_96)
