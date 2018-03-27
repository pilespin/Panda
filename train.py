
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import regularizers

from KerasHelper import KerasHelper

import keras as k
import numpy as np

import data


def main():

	KerasHelper.log_level_decrease()

	(X_train, Y_train), (X_test, Y_test) = data.load_data()
	Y_train = to_categorical(Y_train, num_classes=2)
	Y_test = to_categorical(Y_test, num_classes=2)
	print(X_train)
	# exit(0)

	k.initializers.Ones()
	inputSize = len(np.array(X_train)[0])
	outputSize = len(np.array(Y_train)[0])
	hiddenSize = inputSize + 10

	model = Sequential()
	model.add(Dense(hiddenSize, input_dim=inputSize, activation='tanh'
		, kernel_regularizer=regularizers.l1_l2(0.01)))
	model.add(Dense(hiddenSize, activation='tanh', kernel_regularizer=regularizers.l1_l2(0.01)))
	model.add(Dense(outputSize, activation='relu'))

	model.compile(loss='binary_crossentropy'
				, optimizer=optimizers.Adam(lr=0.001, decay=0.001)
				, metrics=['accuracy'])

	for i in range(1000):
		model.fit(X_train, Y_train
					, epochs=1000
					, batch_size=128
					, validation_data=(X_test, Y_test))

		KerasHelper.save_model(model, "model")


if __name__ == '__main__':
	main()
