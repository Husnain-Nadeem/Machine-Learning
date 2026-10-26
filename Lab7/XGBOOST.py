#STACKING CODE 
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
base_madels = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42))
    ('lr', LogisticRegression(max_iter=1000)),
]

stacl_clf = StackingClassifier(
    estimators=base_madels,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
stacl_clf.fit(X_train_num,y_train_prepared)
print(accuracy(stacl_clf,X_train_num,y_train_prepared))

X_test = test_data.drop ('class', axis =1)
y_test = test_data['class']
y_train_prepared= label_enc.transform(y_test)