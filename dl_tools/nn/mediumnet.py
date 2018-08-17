from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda


class MediumNet:
    @staticmethod
    def build(config, classes, softmax=True, scale_adjust_wb=None):
        model = Sequential()
        input_shape = config.input_shape
        chan_dim = -1

        # 1. CONV => RELU => BN => POOL
        model.add(Conv2D(48, (9, 9), strides=(5, 5), input_shape=input_shape))
        model.add(Activation("relu"))
        #model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 2. CONV => RELU => BN => POOL
        model.add(Conv2D(96, (3, 3)))
        model.add(Activation("relu"))
        #model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 1. FC => RELU => BN => FC
        model.add(Flatten())
        model.add(Activation("relu"))
        #model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(128))
        model.add(Activation("relu"))
        #model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(16))
        model.add(Activation("relu"))

        # final layer
        model.add(Dense(classes))
        if softmax:
            model.add(Activation("softmax"))
        if scale_adjust_wb is not None:
            model.add(Lambda(lambda x: scale_adjust_wb[0] * x + scale_adjust_wb[1]))

        # return the constructed network architecture
        return model
