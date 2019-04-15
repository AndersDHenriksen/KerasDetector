from keras import backend as K


def get_all_layer_outputs(model, test_stack, learning_phase=0):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers if 'input' not in layer.name]          # all layer outputs
    functor = K.function([inp, K.learning_phase()], outputs)    # evaluation function

    # Testing
    layer_outs = functor([test_stack, learning_phase])
    return layer_outs