from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Reshape, Lambda, Flatten, Dense
from keras.regularizers import l2
import keras.backend as K
import numpy as np
from dl_tools.nn.nn_tools import set_regularization


class CenterOfMass:
    def __init__(self, input_shape, scale_factors_uv):
        u_cor, v_cor = np.meshgrid(np.arange(input_shape[1], dtype=np.float), np.arange(input_shape[0], dtype=np.float))
        self._u_cor = u_cor
        self._v_cor = v_cor
        self._scale_factors_uv = scale_factors_uv

    @property
    def __name__(self):
        return 'center_of_mass'

    def __call__(self, image_stack):
        image_sum = K.sum(image_stack, axis=(1, 2)) + 1e-9
        com_u = self._scale_factors_uv[0] * K.sum(self._u_cor * image_stack, axis=(1, 2)) / image_sum
        com_v = self._scale_factors_uv[1] * K.sum(self._v_cor * image_stack, axis=(1, 2)) / image_sum
        return K.stack([com_u, com_v], axis=1)


class ConvTransform:
    @staticmethod
    def build(config):

        model = Sequential()
        input_shape = config.input_shape

        # 1. CONV
        model.add(Conv2D(8, (7, 7), input_shape=input_shape, padding="same"))
        model.add(Activation("relu"))

        # Size reduction
        model.add(Conv2D(16, (7, 7), padding="same"))
        model.add(Activation("relu"))

        # 2. CONV
        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation("relu"))

        # 3. CONV
        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation("relu"))

        # 5. CONV, 1x1
        model.add(Conv2D(12, (1, 1)))
        model.add(Activation("relu"))
        model.add(Dropout(0.3))

        # 6. CONV, 1x1
        model.add(Conv2D(1, (1, 1)))
        model.add(Activation("relu"))

        # Add regularization
        model = set_regularization(model, kernel_regularizer=l2(config.l2))  # custom_objects={'center_of_mass': center_of_mass}

        final_image_shape = [s for s in input_shape[:2]]
        model.add(Reshape(final_image_shape))
        center_of_mass = CenterOfMass(final_image_shape, scale_factors_uv=(1, 1))
        model.add(Lambda(center_of_mass, output_shape=(2,), trainable=False))

        # return the constructed network architecture
        return model
