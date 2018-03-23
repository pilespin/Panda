
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from KerasHelper import KerasHelper

import pandas as pd
import numpy as np
# import os


def load_data():

    train_df = pd.read_csv('datasets/train.csv')
    # test_df = pd.read_csv('datasets/test.csv')

    # tmp = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


    train_df.Cabin = train_df.Cabin.fillna('N')
    train_df.Cabin = train_df.Cabin.apply(lambda x: x[0])
    train_df.Cabin = train_df.Cabin.map( {'N': -1, 'C': 0, 'E': 2, 'G': 3, 'D': 4, 'A': 5, 'B': 6, 'F': 7, 'T': 8,} ).astype(float)
    train_df.Embarked = train_df.Embarked.map( {'C': -1, 'Q': 0, 'S': 1} ).astype(float)
    train_df.Sex = train_df.Sex.map( {'female':-1, 'male': 1} ).astype(float)

    for dataset in [train_df]:

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

        dataset['IsAlone'] = 0
        dataset['IsTwo'] = 0
        dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1
        dataset.loc[dataset['Family'] == 2, 'IsTwo'] = 1

    train_df = train_df.dropna(axis=0, how='any')

    remove = ['PassengerId', 'Name', 'Ticket']
    train_df = train_df.drop(remove, axis=1)


    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]

    split = int(len(X_train) * 0.2)

    X_test = X_train[:split]
    X_train = X_train[split:]

    Y_test = Y_train[:split]
    Y_train = Y_train[split:]

    return (X_train, Y_train), (X_test, Y_test)


def main():

	KerasHelper.log_level_decrease()

	(X_train, Y_train), (X_test, Y_test) = load_data()
	Y_train = to_categorical(Y_train, num_classes=2)
	Y_test = to_categorical(Y_test, num_classes=2)


	model = Sequential()
	model.add(Dense(500, input_dim=16, activation='tanh'))
	model.add(Dense(100, activation='tanh'))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='binary_crossentropy'
				, optimizer=optimizers.Adam(lr=0.0003)
				, metrics=['accuracy'])

	model.fit(X_train, Y_train
			, epochs=99999
			, batch_size=100
			, validation_data=(X_test, Y_test))

	# # evaluate the model
	# al = 0
	# for i in range(nb_output):
	# 	scores = model.evaluate(X_train[i], Y_train[i])
	# 	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	# 	al = al + (scores[1]*100)

	# print("\nALL %s: %.2f%%" % (model.metrics_names[1], (al/nb_output)))

	# Evaluate
	# print ("Evaluate:")
	# path = "mnist_png/testing/"
	# (X_eval, Y_eval) = KerasHelper.get_dataset_with_folder(path, 'L')

	# scores = model.evaluate(X_eval, Y_eval)
	# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	# KerasHelper.save_model(model, "model")




if __name__ == '__main__':
	main()
