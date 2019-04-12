from keras.models import model_from_json


def set_regularization(model, custom_objects=None, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None):

    for layer in model.layers:

        # set kernel_regularizer
        if kernel_regularizer is not None and hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = kernel_regularizer

        # set bias_regularizer
        if bias_regularizer is not None and hasattr(layer, 'bias_regularizer'):
            layer.bias_regularizer = bias_regularizer

        # set activity_regularizer
        if activity_regularizer is not None and hasattr(layer, 'activity_regularizer'):
            layer.activity_regularizer = activity_regularizer

    new_model = model_from_json(model.to_json(), custom_objects)
    new_model.set_weights(model.get_weights())

    return new_model