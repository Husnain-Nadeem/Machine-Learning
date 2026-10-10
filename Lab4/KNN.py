import matplotlib.pyplot as plt
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

wine = load_wine()
X ,y = wine.data ,wine.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#Define the k calues to test 
k_values = [1,3,5,7,9]

#Train a kn classifeer

accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

#plot accuracy againt k values

plt.scatter(k_values, accuracy_scores)
plt.xlabel("k ")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs. K's Values")
plt.show()

k_values=range(1,30)

accutacy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)


#pritn the best value of k 

best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print("Best value of k:", best_k)