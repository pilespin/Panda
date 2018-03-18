#!/usr/bin/env python3

# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Figures inline and set visualization style
# %matplotlib inline
sns.set()

# Import data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# View first few lines of training data
# print(train_df.head())
# print(train_df.info())
# print(train_df.describe())

tmp = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# tmp = train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# tmp = train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(tmp)

# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=50)

# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend();

# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()

# plt.show()

train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

train_df = train_df.dropna(axis=0, how='any')
test_df = test_df.dropna(axis=0, how='any')

# for dataset in [train_df, test_df]:
	# dataset['Embarked'] = dataset['Embarked'].fillna('nope')
	# for label in dataset:
		# dataset[label] = dataset[label].map( {'nope': 0} ).astype(int)
		# dataset[label] = dataset[label].fillna('nope')

for dataset in [train_df, test_df]:

	# dataset['TrAge6'] = 0
	# dataset['TrAge18'] = 0
	# dataset['TrAge40'] = 0
	# dataset['TrAge60'] = 0
	# dataset['TrAgeOld'] = 0
	# dataset.loc[dataset['Age'] <= 6, 'TrAge6'] = 1
	# dataset.loc[(dataset['Age'] > 6) & (dataset['Age'] <= 18), 'TrAge18'] = 1
	# dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 40), 'TrAge40'] = 1
	# dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'TrAge40'] = 1
	# dataset.loc[ dataset['Age'] > 60, 'TrAgeOld'] = 1

	dataset['Family'] = (dataset['Parch'] + dataset['SibSp'])
	dataset['Sex'] = dataset['Sex'].map( {'female':-1, 'male': 1} ).astype(float)
	dataset['Embarked'] = dataset['Embarked'].map( {'C': -1, 'Q': 0, 'S': 1} ).astype(float)

	# dataset['IsAlone'] = 0
	# dataset['IsTwo'] = 0
	# dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1
	# dataset.loc[dataset['Family'] == 2, 'IsTwo'] = 1


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

X_valid = X_train[:500]
X_traincut = X_train[500:]

Y_valid = Y_train[:500]
Y_traincut = Y_train[500:]

# print (X_train.shape)
# print (X_valid.shape)
# print (X_traincut.shape)
# exit(0)



X_test  = test_df.copy()

print(X_train.head())
print(Y_train.head())


linear_svc = LinearSVC()
linear_svc.fit(X_traincut, Y_traincut)
# Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_valid, Y_valid) * 100, 2)
# print(acc_linear_svc)

logreg = LogisticRegression()
logreg.fit(X_traincut, Y_traincut)
# Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_valid, Y_valid) * 100, 2)
# print(acc_log)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_traincut, Y_traincut)
# Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_valid, Y_valid) * 100, 2)
# print(acc_knn)

gaussian = GaussianNB()
gaussian.fit(X_traincut, Y_traincut)
# Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_valid, Y_valid) * 100, 2)
# print(acc_gaussian)

svc = SVC()
svc.fit(X_traincut, Y_traincut)
# Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_valid, Y_valid) * 100, 2)
# print(acc_svc)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_traincut, Y_traincut)
# Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_valid, Y_valid) * 100, 2)
# print(acc_decision_tree)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_traincut, Y_traincut)
# Y_pred = random_forest.predict(X_test)
random_forest.score(X_valid, Y_valid)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# print(acc_random_forest)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Linear SVC', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, 
              acc_linear_svc, acc_decision_tree]})
models = models.sort_values(by='Score', ascending=False)
print(models)

# submission = pd.DataFrame({
#         "PassengerId": test_df["PassengerId"],
#         "Survived": Y_pred
#     })
# submission.to_csv('../output/submission.csv', index=False)
