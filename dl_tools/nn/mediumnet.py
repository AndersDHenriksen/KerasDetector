from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Dropout, Flatten
import numpy as np

class MediumNet:
    @staticmethod
    def build(config, classes, softmax=True, scale_adjust_wb=None):
        model = Sequential()
        input_shape = config.input_shape
        chan_dim = -1

        # 1. CONV => RELU => BN => POOL
        model.add(Conv2D(48, (7, 7), strides=(3, 3), input_shape=input_shape))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chan_dim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # 2. CONV => RELU => BN => POOL
        model.add(Conv2D(96, (5, 5), strides=(2, 2)))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chan_dim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # 3. CONV => RELU => BN => POOL
        model.add(Conv2D(128, (3, 3), strides=(2, 2)))
        model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # 4. CONV => RELU => BN => POOL
        model.add(Conv2D(96, (3, 3)))
        model.add(Activation("relu"))

        # 5. CONV => RELU => BN => POOL
        model.add(Conv2D(64, (1, 1)))
        model.add(Activation("relu"))
        model.add(Dropout(0.3))

        # 1. FC => RELU => BN => FC
        model.add(Flatten())
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(256))
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.6))

        model.add(Dense(16))
        model.add(Activation("relu"))

        # final layer
        model.add(Dense(classes))
        if softmax:
            model.add(Activation("softmax"))
        if scale_adjust_wb is not None:
            # The below line doesn't save/load well as this is a custom object. Thus replaced by dense layer
            # model.add(Lambda(lambda x: scale_adjust_wb[0] * x + scale_adjust_wb[1]))
            input_shape = (None, classes)
            scale_layer = Dense(classes, trainable=False, input_shape=input_shape, )
            scale_layer.build(input_shape=input_shape)
            scale_layer.set_weights([np.diag(scale_adjust_wb[0]), scale_adjust_wb[1]])
            model.add(scale_layer)

        # return the constructed network architecture
        return model
