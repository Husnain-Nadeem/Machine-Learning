
#Gradient Boosting 

from sklearn.ensemble import GradientBoostingClassifier

grad_boost_clf = GradientBoostingClassifier(
    n_estimator=500,
    learning_rate=0.8,
    random_state=42,
    max_depth=2
)
grad_boost_clf.fit(X_train,y_train_prepared)
print(accuracy(grad_boost_clf,X_train,y_train_prepared))

#ADABOOST CODE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
adaboost_clf= AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, class_weight="balance"),
    n_estimators=300,
    learning_rate=0.5
)
adaboost_clf.fit(X_train, y_train_prepared)
print(accuracy(adaboost_clf,X_train,y_train_prepared))
