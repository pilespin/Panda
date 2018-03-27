
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import pandas as pd
import os

import data


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    # print("--------------")
    # print(dataset)
    # print("--------------")
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def main(argv):

    # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    # model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=(784,)))
    # model.add(Dropout(0.2))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(num_classes, activation='softmax'))

    # model.summary()

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=RMSprop(),
    #               metrics=['accuracy'])

    # history = model.fit(x_train, y_train,
    #                     batch_size=batch_size,
    #                     epochs=epochs,
    #                     verbose=1,
    #                     validation_data=(x_test, y_test))
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

############################################################################
    # args = parser.parse_args(argv[1:])

    (X_train, Y_train), (X_test, Y_test) = data.load_data()

    # k = tf.keras

    # model = k.Sequential()
    # model.add(k.layers.Dense(units=64, input_dim=17, activation='relu'))
    # model.add(k.layers.Dense(units=2, activation='softmax'))

    # # model.summary()

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='sgd',
    #               metrics=['accuracy'])

    # model.compile(loss=k.losses.sparse_categorical_crossentropy,
    #               optimizer=k.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True))

    # history = model.fit(X_train, Y_train,
    #                     batch_size=128,
    #                     epochs=50,
    #                     verbose=1,
    #                     validation_data=(X_test, Y_test))
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # print(score)
    # # print('Test loss:', score[0])
    # # print('Test accuracy:', score[1])


    # You can now iterate on your training data in batches:

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    # model.fit(X_train, Y_train, epochs=500)
    # Alternatively, you can feed batches to your model manually:

    # for i in range(20):
    #     model.train_on_batch(X_train, Y_train)
    # # Evaluate your performance in one line:

    # loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=128)
    # # print(loss_and_metrics)
    # # Or generate predictions on new data:

    # # classes = model.predict(X_test, batch_size=128)

############################################ Tensorflow ################################################

    sess = tf.InteractiveSession()

    my_feature_columns = []
    for key in X_train:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[25, 25],
        n_classes=2)

    classifier.train(
        input_fn=lambda:train_input_fn(X_train, Y_train, 128),
        steps=10000)

    # with tf.name_scope('accuracy'):

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(X_test, Y_test, 128))

    print('\nTest set accuracy: {0:0.3f}%\n'.format(eval_result['accuracy']*100))
    tf.summary.scalar('accuracy', eval_result['accuracy'])


    # # Generate predictions from the model
    # expected = ['Setosa', 'Versicolor', 'Virginica']
    # predict_x = {
    #     'SepalLength': [5.1, 5.9, 6.9],
    #     'SepalWidth': [3.3, 3.0, 3.1],
    #     'PetalLength': [1.7, 4.2, 5.4],
    #     'PetalWidth': [0.5, 1.5, 2.1],
    # }

    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(predict_x,
                                            labels=None,
                                            batch_size=args.batch_size))

    # template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    # for pred_dict, expec in zip(predictions, expected):
    #     class_id = pred_dict['class_ids'][0]
    #     probability = pred_dict['probabilities'][class_id]

    #     print(template.format(iris_data.SPECIES[class_id],
    #                           100 * probability, expec))
############################################ Tensorflow ################################################


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run(main)
