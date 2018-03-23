#!/usr/bin/env python3

# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Figures inline and set visualization style
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

remove = ['PassengerId', 'Name', 'Ticket']

train_df = train_df.drop(remove, axis=1)
test_df = test_df.drop(remove, axis=1)

train_df.Cabin = train_df.Cabin.fillna('N')
train_df.Cabin = train_df.Cabin.apply(lambda x: x[0])
train_df.Cabin = train_df.Cabin.map( {'N': -1, 'C': 0, 'E': 2, 'G': 3, 'D': 4, 'A': 5, 'B': 6, 'F': 7, 'T': 8,} ).astype(float)

# print (train_df.shape)
# print(train_df.Cabin.unique())
# exit(0)

for dataset in [train_df, test_df]:

	dataset['TrAge6'] = 0
	dataset['TrAge18'] = 0
	dataset['TrAge40'] = 0
	dataset['TrAge60'] = 0
	dataset['TrAgeOld'] = 0
	dataset.loc[dataset['Age'] <= 6, 'TrAge6'] = 1
	dataset.loc[(dataset['Age'] > 6) & (dataset['Age'] <= 18), 'TrAge18'] = 1
	dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 40), 'TrAge40'] = 1
	dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'TrAge40'] = 1
	dataset.loc[ dataset['Age'] > 60, 'TrAgeOld'] = 1

	dataset['Family'] = (dataset['Parch'] + dataset['SibSp'])
	dataset['Sex'] = dataset['Sex'].map( {'female':-1, 'male': 1} ).astype(float)
	dataset['Embarked'] = dataset['Embarked'].map( {'C': -1, 'Q': 0, 'S': 1} ).astype(float)

	dataset['IsAlone'] = 0
	dataset['IsTwo'] = 0
	dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1
	dataset.loc[dataset['Family'] == 2, 'IsTwo'] = 1

train_df = train_df.dropna(axis=0, how='any')
test_df = test_df.dropna(axis=0, how='any')



X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]



# val_split = 0.2
# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=val_split)

split = int(len(X_train) * 0.2)

X_test = X_train[:split]
X_train = X_train[split:]

Y_test = Y_train[:split]
Y_train = Y_train[split:]

scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test)

print (X_train.shape)
print (X_test.shape)

# print(X_train)
# print(Y_train.head())


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
acc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 2)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc_log = round(logreg.score(X_test, Y_test) * 100, 2)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_test, Y_test) * 100, 2)

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 2)

svc = SVC()
svc.fit(X_train, Y_train)
acc_svc = round(svc.score(X_test, Y_test) * 100, 2)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)

mlp = MLPClassifier(max_iter=2000, activation='tanh', solver='lbfgs', hidden_layer_sizes=(64,64,64, 10), random_state=1)
# mlp = MLPClassifier(activation='tanh', alpha=1e-05, batch_size=64,
#        beta_1=0.9, beta_2=0.999, early_stopping=False,
#        epsilon=1e-08, hidden_layer_sizes=(25, 25, 10), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=2000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#        warm_start=False)

mlp.fit(X_train, Y_train)
acc_mlp = round(mlp.score(X_test, Y_test) * 100, 2)
# print(acc_mlp)


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Linear SVC', 'Decision Tree', 'MLP'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, 
              acc_linear_svc, acc_decision_tree, acc_mlp]})
models = models.sort_values(by='Score', ascending=False)
print(models)

# submission = pd.DataFrame({
#         "PassengerId": test_df["PassengerId"],
#         "Survived": Y_pred
#     })
# submission.to_csv('../output/submission.csv', index=False)
