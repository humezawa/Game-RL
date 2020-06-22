import numpy as np
import math
from keras import models, layers, optimizers, losses, metrics, activations, regularizers, callbacks
import random
from buffer import ILBuffer

BATCH_SIZE_TEST = math.floor(750 * 0.8)
BATCH_SIZE_VAL = math.floor(750 * 0.2)
MINI_BATCH_SIZE = 32
EPOCHS = 20
VAL_SPLIT = 0.0
LAMB = 0.0
LR = 0.001


def mux(input, n):
    output = np.zeros((input.shape[0], n), dtype=np.bool)
    for i in range(0, input.shape[0]):
        output[i, input[i, 0]] = 1

    return output


def scheduler(epoch, lr):
    print(lr)
    return LR


class CNN:
    """
    Represents a Deep Q-Networks (DQN) agent.
    """

    def __init__(self, learning_rate=0.001, file_code=345):
        """
        Creates a Deep Q-Networks (DQN) agent.

        :param learning_rate: learning rate of the action-value neural network.
        :type learning_rate: float.

        """
        self.il_buffer = ILBuffer()
        self.il_buffer.il_load(file_code=file_code)
        self.learning_rate = learning_rate
        self.model = self.make_model()
        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.Adam(),
                           metrics=["categorical_accuracy"])
        self.lr = 0.001

    def make_model(self):
        """
        Makes the action-value neural network model using Keras.

        :return: action-value neural network.
        :rtype: Keras' model.
        """

        model = models.Sequential()

        # Todo: implement cnn

        model.add(layers.Conv2D(filters=3,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.relu,
                                name='chanel_reducer1',
                                input_shape=(800, 600, 12),
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.Conv2D(filters=4,
                                kernel_size=(5, 5),
                                strides=(1, 1),
                                activation=activations.relu,
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.AveragePooling2D(pool_size=(2, 2),
                                          strides=(2, 2)))

        model.add(layers.Conv2D(filters=3,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.relu,
                                name='chanel_reducer2',
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.Conv2D(filters=8,
                                kernel_size=(5, 5),
                                strides=(1, 1),
                                activation=activations.relu,
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.AveragePooling2D(pool_size=(2, 2),
                                          strides=(2, 2)))

        model.add(layers.Conv2D(filters=3,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.relu,
                                name='chanel_reducer3',
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.Conv2D(filters=16,
                                kernel_size=(5, 5),
                                strides=(1, 1),
                                activation=activations.relu,
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.AveragePooling2D(pool_size=(2, 2),
                                          strides=(2, 2)))

        model.add(layers.Conv2D(filters=3,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.relu,
                                name='chanel_reducer4',
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.Conv2D(filters=32,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                activation=activations.relu,
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.AveragePooling2D(pool_size=(2, 2),
                                          strides=(2, 2)))

        model.add(layers.Conv2D(filters=3,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.relu,
                                name='chanel_reducer5',
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.Conv2D(filters=64,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                activation=activations.relu,
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.AveragePooling2D(pool_size=(2, 2),
                                          strides=(2, 2)))

        model.add(layers.Conv2D(filters=3,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.relu,
                                name='chanel_reducer6',
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.Conv2D(filters=128,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                activation=activations.relu,
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.AveragePooling2D(pool_size=(2, 2),
                                          strides=(2, 2)))

        model.add(layers.Conv2D(filters=1,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.relu,
                                name='chanel_reducer7',
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.Flatten())

        model.add(layers.Dense(units=64,
                               activation=activations.relu,
                                kernel_regularizer=regularizers.l2(LAMB)))

        model.add(layers.Dense(units=16,
                               activation=activations.softmax))

        print(model.summary())

        return model

    def act(self, state):
        """
        :param state: 4 following frames
        :type state: numpy ndarray of shape(x,800,600,12)
        :return action: index which represents a action
        :type action: numpy ndarray of shape (x, 1)
        """

        action = self.model.predict(state)
        return action

    def train(self, epochs):
        # init variables
        loss_hist = []
        acc_list = []

        # predicting that we will have to move every image we take, each step is being made separated
        for epoch in range(0, epochs):
            print('epoch {}/{}'.format(epoch + 1, epochs))
            # each epoch the learning rate is reduced
            self.lr = 0.001 * (0.7 ** (epoch // 20))
            # take a random test sample already multiplied by the mover
            states_sample, actions_sample = self.il_buffer.il_extended_mini_batch(BATCH_SIZE_TEST, 'test')
            # encrypt each number using a mux
            actions_sample = mux(actions_sample, 16)

            # uses callback to update the learning rate according to self.__scheduler method
            callback = callbacks.LearningRateScheduler(self.__scheduler)
            hist = self.model.fit(x=states_sample,
                                  y=actions_sample,
                                  epochs=1,
                                  verbose=1,
                                  callbacks=[callback],
                                  batch_size=MINI_BATCH_SIZE)
            loss_hist = loss_hist + hist.history['loss']
            if epoch % 10 == 9:
                states_sample, actions_sample = self.il_buffer.il_extended_mini_batch(BATCH_SIZE_VAL, 'validation')
                acc_list.append(self.__evaluate(states_sample, actions_sample)[1])

        return [loss_hist, acc_list]

    def __scheduler(self, epoch, lr):
        return self.lr

    # not ok
    def __evaluate(self, states_sample, actions_sample):
        # init variables

        actions_sample = mux(actions_sample, 16)

        hist = self.model.evaluate(x=states_sample, y=actions_sample, batch_size=MINI_BATCH_SIZE)

        return hist

    def load(self, name):
        """
        Loads the neural network's weights from disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.load_weights(name + 'h5')

    def save(self, name):
        """
        Saves the neural network's weights to disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.save_weights(name + '.h5')


'''
model.add(layers.AveragePooling2D(pool_size=(3, 3),
                                          strides=(3, 3),
                                          input_shape=(800, 600, 12),
                                          name='quality_reducer'))

        model.add(layers.Conv2D(filters=3,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.tanh,
                                name='chanel_reducer1'))

        model.add(layers.Conv2D(filters=6,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                activation=activations.tanh))

        model.add(layers.AveragePooling2D(pool_size=(3, 3),
                                          strides=(3, 3)))

        model.add(layers.Conv2D(filters=1,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.tanh,
                                name='chanel_reducer2'))

        model.add(layers.Conv2D(filters=6,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                activation=activations.tanh))

        model.add(layers.AveragePooling2D(pool_size=(3, 3),
                                          strides=(3, 3)))

        model.add(layers.Conv2D(filters=1,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.tanh,
                                name='chanel_reducer3'))

        model.add(layers.Conv2D(filters=6,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                activation=activations.tanh))

        model.add(layers.AveragePooling2D(pool_size=(3, 3),
                                          strides=(3, 3)))

        model.add(layers.Conv2D(filters=1,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                activation=activations.tanh,
                                name='chanel_reducer4'))

        model.add(layers.Conv2D(filters=6,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                activation=activations.tanh))

        model.add(layers.AveragePooling2D(pool_size=(3, 3),
                                          strides=(3, 3)))

        model.add(layers.Flatten())

        model.add(layers.Dense(units=16,
                               activation=activations.softmax))
'''
