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

from sklearn.ensemble import ExtraTreesClassifier

import data

(X_train, Y_train), (X_test, Y_test) = data.load_data()

np.set_printoptions(threshold='nan', linewidth=200)
np.set_printoptions(suppress=True)

model = ExtraTreesClassifier()
model.fit(X_train, Y_train)
a = np.array(model.feature_importances_)
print(a)

# linear_svc = LinearSVC()
# linear_svc.fit(X_train, Y_train)
# acc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 2)

# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# acc_log = round(logreg.score(X_test, Y_test) * 100, 2)

# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(X_train, Y_train)
# acc_knn = round(knn.score(X_test, Y_test) * 100, 2)

# gaussian = GaussianNB()
# gaussian.fit(X_train, Y_train)
# acc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 2)

# svc = SVC()
# svc.fit(X_train, Y_train)
# acc_svc = round(svc.score(X_test, Y_test) * 100, 2)

# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, Y_train)
# acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)

# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
# acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)

# mlp = MLPClassifier(max_iter=2000, activation='tanh', solver='lbfgs', hidden_layer_sizes=(100,100))
# # mlp = MLPClassifier(activation='tanh', alpha=1e-05, batch_size=64,
# #        beta_1=0.9, beta_2=0.999, early_stopping=False,
# #        epsilon=1e-08, hidden_layer_sizes=(25, 25, 10), learning_rate='constant',
# #        learning_rate_init=0.001, max_iter=2000, momentum=0.9,
# #        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
# #        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
# #        warm_start=False)

# mlp.fit(X_train, Y_train)
# acc_mlp = round(mlp.score(X_test, Y_test) * 100, 2)
# # print(acc_mlp)


# models = pd.DataFrame({
#     'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
#               'Random Forest', 'Naive Bayes', 
#               'Linear SVC', 'Decision Tree', 'MLP'],
#     'Score': [acc_svc, acc_knn, acc_log, 
#               acc_random_forest, acc_gaussian, 
#               acc_linear_svc, acc_decision_tree, acc_mlp]})
# models = models.sort_values(by='Score', ascending=False)
# print(models)

# submission = pd.DataFrame({
#         "PassengerId": test_df["PassengerId"],
#         "Survived": Y_pred
#     })
# submission.to_csv('../output/submission.csv', index=False)
