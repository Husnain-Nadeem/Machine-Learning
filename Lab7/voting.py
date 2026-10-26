
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt 


import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_openml

elec_data = fetch_openml(name='electricity', version=1)
elec_df = elec_data.frame

print("Data shape:", elec_data.data.shape)
print("Feature names:", elec_data.feature_names)
print("Target names:", elec_data.target_names)


from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    elec_df, test_size=0.25, random_state=20
)

print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))

train_data.info()
train_data.describe()
train_data.isnull().sum()
corr = train_data.corr(numeric_only=True)

plt.figure(figsize=(12, 7))
sns.heatmap(corr, annot=True, cmap='crest')
plt.show()

plt.figure(figsize=(12, 7))
sns.countplot(data=train_data, x='class')
plt.show()

plt.figure(figsize=(13, 8))
sns.countplot(data=train_data, x='day', hue='class')
plt.show()

train_data.hist(bins=50, figsize=(15, 10))
plt.show()
from sklearn.preprocessing import LabelEncoder

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']

label_enc = LabelEncoder()
y_train_prepared = label_enc.fit_transform(y_train)

print(label_enc.classes_)


from sklearn.metrics import accuracy_score

def accuracy(model, data, labels):
    predictions = model.predict(data)
    acc = accuracy_score(labels, predictions)

    return acc
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X_train_num = X_train_copy()
X_train_num['day'] = X_train_num['day'].astype(int)
models = [('lr', LogisticRegression(max_iter=1000)),
          ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
          ('svc', SVC(probability=True, random_state=42))]

voting_hard = VotingClassifier(models, voting='hard')
voting_soft = VotingClassifier(models, voting='soft')

voting_hard.fit(X_train_num, y_train_prepared)
voting_soft.fit(X_train_num, y_train_prepared)
print(accuracy(voting_hard,X_train_num,y_train_prepared))
print(accuracy(voting_soft, X_train_num, y_train_prepared))

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_classirfier = BaggingClassifier(
    DecisionTreeClassifier(class_weight='balanced'),
    max_samples=0.5 , max_features=0.5 , bootstrap=False
)

bag_classirfier.fit(X_train, y_train_prepared)
print(accuracy(bag_classirfier,X_train,y_train_prepared))


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

#XGBOOST
import xqboost as xqb
xqb_clf =xqb.XGClassifier()
xqb_clf.fit(X_train_num, y_train_prepared)
print(accuracy(xqb_clf,X_train_num,y_train_prepared))

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