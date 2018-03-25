
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import regularizers

from KerasHelper import KerasHelper

import pandas as pd
import keras as k
import numpy as np

from sklearn import preprocessing


def load_data():
	le = preprocessing.LabelEncoder()
	# lb = preprocessing.LabelBinarizer()

	train_df = pd.read_csv('datasets/train.csv')
	# test_df = pd.read_csv('datasets/test.csv')

	# tmp = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


	# train_df.Cabin = train_df.Cabin.fillna('N')
	# train_df.Embarked = train_df.Embarked.fillna('Nan')
	# train_df.Age = train_df.Age.fillna(-1)
	# train_df.Cabin = train_df.Cabin.apply(lambda x: x[0])
	# train_df.Cabin = train_df.Cabin.map( {'N': -1, 'C': 0, 'E': 2, 'G': 3, 'D': 4, 'A': 5, 'B': 6, 'F': 7, 'T': 8,} ).astype(float)
	# train_df.Embarked = train_df.Embarked.map( {'N' : -1,'C': 0, 'Q': 1, 'S': 2} ).astype(float)
	# train_df.Sex = train_df.Sex.map( {'female':-1, 'male': 1} ).astype(float)




	for dataset in [train_df]:

		dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
		dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
		dataset['Cabin'].fillna(dataset['Cabin'].mode()[0], inplace=True)


		dataset['Sex'] = le.fit_transform(dataset['Sex'])
		dataset['Embarked'] = le.fit_transform(dataset['Embarked'])
		dataset['Cabin'] = le.fit_transform(dataset['Cabin'])
		dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
		dataset['Title'] = le.fit_transform(dataset['Title'])
		# dataset['CabinNum'] = dataset['Cabin'][1:]
		# dataset['CabinNum'].fillna(dataset['CabinNum'].mode(), inplace=True)


		dataset['TrAge6'] = -1
		dataset['TrAge18'] = -1
		dataset['TrAge40'] = -1
		dataset['TrAge60'] = -1
		dataset['TrAgeOld'] = -1
		dataset.loc[dataset['Age'] <= 6, 'TrAge6'] = 1
		dataset.loc[(dataset['Age'] > 6) & (dataset['Age'] <= 18), 'TrAge18'] = 1
		dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 40), 'TrAge40'] = 1
		dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'TrAge40'] = 1
		dataset.loc[ dataset['Age'] > 60, 'TrAgeOld'] = 1

		print(pd.get_dummies(dataset['Age']))
		exit(0)

		dataset['Family'] = (dataset['Parch'] + dataset['SibSp'])

		dataset['IsAlone'] = -1
		dataset['IsTwo'] = -1
		dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1
		dataset.loc[dataset['Family'] == 2, 'IsTwo'] = 1

	# train_df = train_df.dropna(axis=0, how='any')

	# remove = ['PassengerId', 'Name', 'Pclass' , 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
	# remove = ['PassengerId', 'Age', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
	remove = ['PassengerId', 'Name', 'Ticket']
	# remove = []
	# remove = ['PassengerId', 'Name', 'Ticket']
	train_df = train_df.drop(remove, axis=1)

	# Check where is null
	# null_columns=train_df.columns[train_df.isnull().any()]
	# print(train_df[null_columns].isnull().sum())
	# exit(0)



	X_train = train_df.drop("Survived", axis=1)
	Y_train = train_df["Survived"]

	print(X_train[:10].to_string())
	# exit(0)

	############################## Scale ##############################
	X_train = preprocessing.MinMaxScaler().fit_transform(X_train.values)

	############################## Split ##############################

	split = int(len(X_train) * 0.2)

	X_test = X_train[:split]
	X_train = X_train[split:]

	Y_test = Y_train[:split]
	Y_train = Y_train[split:]

	# print(X_train.shape)
	# print(X_test.shape)
	# exit(0)

	return (X_train, Y_train), (X_test, Y_test)


def main():

	KerasHelper.log_level_decrease()

	(X_train, Y_train), (X_test, Y_test) = load_data()
	Y_train = to_categorical(Y_train, num_classes=2)
	Y_test = to_categorical(Y_test, num_classes=2)
	print(X_train)
	# exit(0)

	k.initializers.Ones()
	inputSize = len(np.array(X_train)[0])
	outputSize = len(np.array(Y_train)[0])
	hiddenSize = inputSize +2

	model = Sequential()
	model.add(Dense(hiddenSize, input_dim=inputSize, activation='tanh', kernel_regularizer=regularizers.l1_l2(0.1)))
	model.add(Dense(hiddenSize, activation='relu'))
	model.add(Dense(outputSize, activation='relu'))

	model.compile(loss='binary_crossentropy'
				, optimizer=optimizers.Adamax(lr=0.01, decay=0.001)
				, metrics=['accuracy'])

	model.fit(X_train, Y_train
				, epochs=99999
				, batch_size=128
				, validation_data=(X_test, Y_test))

	# # evaluate the model
	# al = 0
	# for i in range(nb_output):
	#  scores = model.evaluate(X_train[i], Y_train[i])
	#  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	#  al = al + (scores[1]*100)

	# print("\nALL %s: %.2f%%" % (model.metrics_names[1], (al/nb_output)))

	# Evaluate
	# print ("Evaluate:")
	# path = "mnist_png/testing/"
	# (X_eval, Y_eval) = KerasHelper.get_dataset_with_folder(path, 'L')

	# scores = model.evaluate(X_eval, Y_eval)
	# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	# KerasHelper.save_model(model, "model")




# if __name__ == '__main__':
main()
