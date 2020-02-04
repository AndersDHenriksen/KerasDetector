import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.callbacks import GifGenerator
from tf_keras_vis.utils.callbacks import Print
from tensorflow.keras.models import load_model

MODEL_PATH = r"C:\NN\Experiments\DropletDetection\2020-01-29 07-55-29 - MediumNet_run2\checkpoint\best_epoch-00023_val-loss-0.01.hdf5"
class_to_maximize = 0

model = load_model(MODEL_PATH)

# Define modifier to replace a softmax function of the last layer to a linear function.
def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear

# Create Activation Maximization object
activation_maximization = ActivationMaximization(model, model_modifier)

loss = lambda x: K.mean(x[:, class_to_maximize])

activation = activation_maximization(loss, steps=512, callbacks=[Print(interval=100), GifGenerator('activation_maximization')])
image = activation[0].astype(np.uint8)

plt.figure()
plt.imshow(image)

