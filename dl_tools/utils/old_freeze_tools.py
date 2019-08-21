import re
from pathlib import Path
from keras import backend as K
from keras.models import Model
from keras.layers import MaxPool2D, Reshape
import tensorflow as tf

def change_layer(model, layer_regex, new_layer, method='replace', additional_layers=None):
    if new_layer is None:
        method = 'delete'
    assert method in ['after', 'before', 'replace', 'delete']
    assert additional_layers is None or isinstance(additional_layers, list)

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update({layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update({model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if 0 and re.match(layer_regex, layer.name):
            if method == 'delete':
                network_dict['new_output_tensor_of'].update({layer.name: x})
                print('Layer {} deleted'.format(layer.name))
                continue
            if method == 'replace':
                x = layer_input
            elif method == 'after':
                x = layer(layer_input)
            elif method == 'before':
                pass

            x = new_layer(x)
            print('Layer {} inserted after layer {}'.format(new_layer.name, layer_input.name))
            if method == 'before':
                x = layer(x)
            if additional_layers is not None:
                for additional_layer in additional_layers:
                    x = additional_layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)


def improve_model(model):
    from keras.layers import Reshape
    from keras.layers.pooling import MaxPool2D

    for layer in model.layers:
        if 'dropout' in layer.name or 'lambda' in layer.name:
            improved_model = change_layer(model, layer.name, None, method='delete')
            return improved_model, True
        if 'flatten' in layer.name:
            new_layer = Reshape(layer.output_shape[1:])
            improved_model = change_layer(model, layer.name, new_layer, method='replace')
            return improved_model, True
        if 'global_max_pooling2d' in layer.name:
            new_layer = MaxPool2D(layer.input_shape[1:3])
            next_layer = Reshape(layer.output_shape[1:])
            improved_model = change_layer(model, layer.name, new_layer, method='replace', additional_layers=[next_layer])
            return improved_model, True
    return model, False


def augment_for_ocv(model):
    model_changing = True
    while model_changing:
        model, model_changing = improve_model(model)
    return model


def save_frozen_protobuf(save_path, session, keep_var_names=None, output_names=None, clear_devices=True):
    if isinstance(save_path, str):
        save_path = Path(save_path)
    K.set_learning_phase(0)
    K.set_image_data_format('channels_last')
    frozen_graph = freeze_session(session, keep_var_names=keep_var_names,
                                  output_names=output_names, clear_devices=clear_devices)
    tf.train.write_graph(frozen_graph, str(save_path.parent), str(save_path.name), as_text=False)


# From: https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """

    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def(add_shapes=True)
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        make_cv2_compatible(frozen_graph)
        return frozen_graph

# -------------------------------------------------------------------------------------------------------------------- #
# from https://stackoverflow.com/questions/49794023/use-keras-model-with-flatten-layer-inside-opencv-3/49817506#49817506
import tensorflow as tf
from tensorflow.core import framework


def find_all_nodes(graph_def, **kwargs):
    for node in graph_def.node:
        for key, value in kwargs.items():
            if getattr(node, key) != value:
                break
        else:
            yield node
    raise StopIteration


def find_node(graph_def, **kwargs):
    try:
        return next(find_all_nodes(graph_def, **kwargs))
    except StopIteration:
        raise ValueError(
            'no node with attributes: {}'.format(
                ', '.join("'{}': {}".format(k, v) for k, v in kwargs.items())))


def walk_node_ancestors(graph_def, node_def, exclude=set()):
    openlist = list(node_def.input)
    closelist = set()
    while openlist:
        name = openlist.pop()
        if name not in exclude:
            node = find_node(graph_def, name=name)
            openlist += list(node.input)
            closelist.add(name)
    return closelist


def remove_nodes_by_name(graph_def, node_names):
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].name in node_names:
            del graph_def.node[i]


def make_shape_node_const(node_def, tensor_values):
    node_def.op = 'Const'
    node_def.ClearField('input')
    node_def.attr.clear()
    node_def.attr['dtype'].type = framework.types_pb2.DT_INT32
    tensor = node_def.attr['value'].tensor
    tensor.dtype = framework.types_pb2.DT_INT32
    tensor.tensor_shape.dim.add()
    tensor.tensor_shape.dim[0].size = len(tensor_values)
    for value in tensor_values:
        tensor.tensor_content += value.to_bytes(4, 'little')
    output_shape = node_def.attr['_output_shapes']
    output_shape.list.shape.add()
    output_shape.list.shape[0].dim.add()
    output_shape.list.shape[0].dim[0].size = len(tensor_values)


def make_cv2_compatible(graph_def):
    # A reshape node needs a shape node as its second input to know how it
    # should reshape its input tensor.
    # When exporting a model using Keras, this shape node is computed
    # dynamically using `Shape`, `StridedSlice` and `Pack` operators.
    # Unfortunately those operators are not supported yet by the OpenCV API.
    # The goal here is to remove all those unsupported nodes and hard-code the
    # shape layer as a const tensor instead.
    for reshape_node in find_all_nodes(graph_def, op='Reshape'):

        # Get a reference to the shape node
        shape_node = find_node(graph_def, name=reshape_node.input[1])

        # Find and remove all unsupported nodes
        garbage_nodes = walk_node_ancestors(graph_def, shape_node, exclude=[reshape_node.input[0]])
        remove_nodes_by_name(graph_def, garbage_nodes)

        # Infer the shape tensor from the reshape output tensor shape
        if not '_output_shapes' in reshape_node.attr:
            raise AttributeError(
                'cannot infer the shape node value from the reshape node. '
                'Please set the `add_shapes` argument to `True` when calling '
                'the `Session.graph.as_graph_def` method.')
        output_shape = reshape_node.attr['_output_shapes'].list.shape[0]
        output_shape = [dim.size for dim in output_shape.dim]

        # Hard-code the inferred shape in the shape node
        make_shape_node_const(shape_node, output_shape[1:])

# -------------------------------------------------------------------------------------------------------------------- #


# Code below is alternative method which might work better if multiple outputs

# From https://github.com/amir-abdi/keras_to_tensorflow
def keras_to_tensorflow(num_output=1, quantize=False, input_fld=".", output_fld=".",
                        input_model_file='final_model.hdf5', output_model_file="", output_node_prefix="output_node"):
    """
    Input arguments:

    num_output: this value has nothing to do with the number of classes, batch_size, etc.,
    and it is mostly equal to 1. If the network is a **multi-stream network**
    (forked network with multiple outputs), set the value to the number of outputs.

    quantize: if set to True, use the quantize feature of Tensorflow
    (https://www.tensorflow.org/performance/quantization) [default: False]

    input_fld: directory holding the keras weights file [default: .]

    output_fld: destination directory to save the tensorflow files [default: .]

    input_model_file: name of the input weight file [default: 'model.h5']

    output_model_file: name of the output weight file [default: args.input_model_file + '.pb']

    output_node_prefix: the prefix to use for output nodes. [default: output_node]

    """

    # initialize
    from keras.models import load_model
    import tensorflow as tf
    from pathlib import Path
    from keras import backend as K

    output_fld = input_fld if output_fld == '' else output_fld
    if output_model_file == '':
        output_model_file = str(Path(input_model_file).name) + '.pb'
    Path(output_fld).mkdir(parents=True, exist_ok=True)
    weight_file_path = str(Path(input_fld) / input_model_file)

    K.set_learning_phase(0)
    K.set_image_data_format('channels_last')

    # Load keras model and rename output
    try:
        net_model = load_model(weight_file_path)
    except ValueError as err:
        print('''Input file specified ({}) only holds the weights, and not the model definition.
        Save the model using mode.save(filename.h5) which will contain the network architecture
        as well as its weights. 
        If the model is saved using model.save_weights(filename.h5), the model architecture is 
        expected to be saved separately in a json format and loaded prior to loading the weights.
        Check the keras documentation for more details (https://keras.io/getting-started/faq/)'''
              .format(weight_file_path))
        raise err
    pred = [None] * num_output
    pred_node_names = [None] * num_output
    for i in range(num_output):
        pred_node_names[i] = output_node_prefix + str(i)
        pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)

    sess = K.get_session()

    # convert variables to constants and save
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    if quantize:
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
        constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_model_file, as_text=False)
    print('saved the freezed graph (ready for inference) at: ', str(Path(output_fld) / output_model_file))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('-input_fld', action="store",
                        dest='input_fld', type=str, default='.')
    parser.add_argument('-output_fld', action="store",
                        dest='output_fld', type=str, default='')
    parser.add_argument('-input_model_file', action="store",
                        dest='input_model_file', type=str, default='model.h5')
    parser.add_argument('-output_model_file', action="store",
                        dest='output_model_file', type=str, default='')
    parser.add_argument('-num_outputs', action="store",
                        dest='num_outputs', type=int, default=1)
    parser.add_argument('-output_node_prefix', action="store",
                        dest='output_node_prefix', type=str, default='output_node')
    parser.add_argument('-quantize', action="store",
                        dest='quantize', type=bool, default=False)
    parser.add_argument('-f')
    args = parser.parse_args()
    parser.print_help()
    print('input args: ', args)

    keras_to_tensorflow(num_output=args.num_outputs,
                        quantize=args.quantize,
                        input_fld=args.input_fld,
                        output_fld=args.output_fld,
                        input_model_file=args.output_model_file,
                        output_model_file=args.output_model_file,
                        output_node_prefix=args.output_node_prefix)