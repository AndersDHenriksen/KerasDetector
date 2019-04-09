from keras.models import Model
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D, Reshape
import keras.backend as K
import numpy as np


class CenterOfMass:
    def __init__(self, input_length):
        self._index = np.arange(input_length, dtype=np.float32)

    @property
    def __name__(self):
        return 'center_of_mass'

    def __call__(self, vector_stack):
        vector_sum = K.sum(vector_stack, axis=1) + 1e-9
        return K.sum(self._index * vector_stack, axis=1) / vector_sum


class ConvTransform:
    @staticmethod
    def build(config):

        input_shape = config.input_shape

        # 1. Conv
        input = Input(input_shape)
        X = Conv2D(8, (7, 7), strides=(1, 1), padding="same", activation='relu')(input)
        X = MaxPooling2D(pool_size=(2, 1))(X)

        # 2. Conv
        X = Conv2D(16, (7, 7), strides=(1, 1), padding="same", activation='relu')(X)
        X = MaxPooling2D(pool_size=(2, 1))(X)

        # 3. Conv
        X = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation='relu')(X)
        X = MaxPooling2D(pool_size=(2, 1))(X)

        # 4. Conv
        X = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation='relu')(X)
        X = MaxPooling2D(pool_size=(2, 1))(X)
        X = Dropout(0.5)(X)

        # 5. CONV, 1x1
        X = Conv2D(12, (5, 1), activation='relu')(X)
        X = Dropout(0.3)(X)

        # 6. CONV, 1x1
        X = Conv2D(1, (1, 1), activation='relu')(X)
        X = Reshape((input_shape[1], ))(X)
        output = Lambda(CenterOfMass(input_shape[1]), output_shape=(1,), trainable=False)(X)

        model = Model(inputs=input, outputs=output)

        # return the constructed network architecture
        return model
