from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import GlobalMaxPooling2D
from keras import regularizers

class SmallNet:
    @staticmethod
    def build(config, classes):
        model = Sequential()
        input_shape = config.input_shape
        r = 0.001 * 30

        # 1. CONV => RELU => BN => POOL
        # model.add(Conv2D(32, (5, 5), strides=(3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(r)))
        model.add(Conv2D(32, (5, 5), input_shape=input_shape, kernel_regularizer=regularizers.l2(r)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(3, (1, 1)))

        # 2. CONV => RELU => BN => POOL
        # model.add(Conv2D(64, (3, 3), strides=(2, 2), kernel_regularizer=regularizers.l2(r)))
        model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(r)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        # model.add(Conv2D(1, (1, 1)))

        # # 3. CONV => RELU => BN => POOL
        model.add(Conv2D(96, (3, 3), kernel_regularizer=regularizers.l2(r)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        #
        # # 4. CONV => RELU => BN => POOL
        # model.add(Conv2D(48, (2, 2)))
        # model.add(Activation("relu"))

        # 1. FC => RELU => BN => FC
        # model.add(Flatten())
        # model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.3))

        # model.add(Dense(64))
        # model.add(Activation("relu"))
        # # model.add(BatchNormalization())
        # model.add(Dropout(0.6))

        model.add(GlobalMaxPooling2D())

        # model.add(Dropout(0.3))
        model.add(Dense(8))
        model.add(Activation("relu"))

        # final layer
        model.add(Dense(classes))
        if classes > 1:
            model.add(Activation("softmax"))
        else:
            model.add(Activation('sigmoid'))

        # return the constructed network architecture
        return model
