from pathlib import Path
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import os


def finalize_for_ocv(model_path):
    K.set_learning_phase(0)

    model = load_model(model_path)
    cv_model = augment_for_ocv(model)
    cv_model_path = model_path.rpslit('.')[0] + '_cv.hdf5'
    cv_model.save(cv_model_path)
    keras_to_opencv(cv_model_path)


def augment_for_ocv(model):
    from kerassurgeon import Surgeon  # pip install kerassurgeon
    from keras.layers import Reshape

    surgeon = Surgeon(model)
    for layer in model.layers:
        if 'dropout' in layer.name or 'lambda' in layer.name:
            surgeon.add_job('delete_layer', layer)
        if 'flatten' in layer.name:
            new_layer = Reshape(layer.output_shape[1:])
            surgeon.add_job('replace_layer', layer, new_layer=new_layer)
    new_model = surgeon.operate()
    return new_model


def keras_to_opencv(model_path):  # Made by Christian, works for OpenCV 4.1. For OpenCV < 4, frozen model seems to work
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
    """
    Prepares Keras dnn model to OpenCV model inference
    :param keras_weights: Path to Keras .h5 file with weights values generated from model.save_weights() 
    :param keras_structure: Path to Keras json file with net structure. Generated from model.to_json()
    :param output: Output path of Tensorflow optimized protobuf file that can be loaded with 
    cv2.dnn.readNetFromTensorflow() 
    :return: None  
    """
    # Sets the learning phase to a fixed value.
    K.set_learning_phase(0)
    net_model = load_model(model_path)
    output_folder = str(Path(model_path).parent)

    # # load keras model
    # with open(keras_structure, 'r') as json_file:
    #     loaded_model_json = json_file.read()  # read json file
    #     net_model = K.models.model_from_json(loaded_model_json)  # create model from json
    #     net_model.load_weights(keras_weights)  # load weights to model

    # Test if successful
    # assert net_model is not None

    # Find input and output node names
    input_nodes_names = [node.name[:-2] for node in net_model.inputs]
    output_nodes_names = [node.name[:-2] for node in net_model.outputs]

    # Freeze graph and convert variables to constants
    sess = K.get_session()
    saver = tf.train.Saver(tf.global_variables())
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_nodes_names)
    # save frozen graph
    frozen_graph_name = "frozen_graph.pb"
    graph_io.write_graph(constant_graph, output_folder, frozen_graph_name, as_text=False)
    print('Saved frozen graph at: ', os.path.join(output_folder, frozen_graph_name))

    # Optimize frozen graph for model inference
    # load frozen graph
    input_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(os.path.join(output_folder, frozen_graph_name), "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    # Optimize graph
    output_graph_def = optimize_for_inference(input_graph_def, input_nodes_names, output_nodes_names, tf.float32.as_datatype_enum)

    # Save inference optimized protobuf file
    # save graph to output file
    optimized_graph_name = "opencv_optimized.pb"
    f = tf.gfile.FastGFile(os.path.join(output_folder, optimized_graph_name), "w")
    f.write(output_graph_def.SerializeToString())
    print('Saved optimized graph (ready for inference) at: ', os.path.join(output_folder, optimized_graph_name))



# ---------------------------------------------- Old currently unused --------------------------------------------------


def save_frozon_protobuf(hdf5_path, clear_devices=True):
    from keras import backend as K
    from keras.models import load_model
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

    if isinstance(hdf5_path, str):
        hdf5_path = Path(hdf5_path)

    # Load keras model
    model = load_model(hdf5_path)
    K.set_learning_phase(0)

    # Freeze graph
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs], clear_devices=clear_devices)

    # Optimize graph
    input_nodes = [model.input.name.rsplit(':')[0]]
    output_nodes = [model.output.name.rsplit(':')[0]]
    optimized_graph = optimize_for_inference(frozen_graph, input_nodes, output_nodes, tf.float32.as_datatype_enum)

    # Save pb file
    tf.train.write_graph(optimized_graph, str(hdf5_path.parent), hdf5_path.stem + '.pb', as_text=False)

    # Prune graph
    graph_def = optimized_graph
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op == 'Const':
            del graph_def.node[i]
        for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim', 'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training', 'Tpaddings']:
            if attr in graph_def.node[i].attr:
                del graph_def.node[i].attr[attr]

    # Save pbtxt file
    tf.train.write_graph(graph_def, str(hdf5_path.parent), hdf5_path.stem + '.pbtxt', as_text=True)




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
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph



def save_frozen_protobuf2(save_path, session, output_names=None, clear_devices=True):
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.tools import optimize_for_inference_lib

    if isinstance(save_path, str):
        save_path = Path(save_path)
    K.set_learning_phase(0)
    K.set_image_data_format('channels_last')

    # fix batch norm nodes. Not working.
    # for node in session.graph_def.node:
    #     if node.op == 'RefSwitch':
    #         node.op = 'Switch'
    #         for index in range(len(node.input)):
    #             if 'moving_' in node.input[index]:
    #                 node.input[index] = node.input[index] + '/read'
    #     elif node.op == 'AssignSub':
    #         node.op = 'Sub'
    #         if 'use_locking' in node.attr:
    #             del node.attr['use_locking']

    # Optimize for inference not working
    # graph_def = optimize_for_inference_lib.optimize_for_inference(session.graph_def, keras_model.input_names,
    #                                                               output_names, tf.float32.as_datatype_enum)
    # session = tf.Session(graph=graph_def)

    tf.train.write_graph(session.graph_def, str(save_path.parent), str(save_path.name), as_text=False)
    tf.train.write_graph(session.graph_def, str(save_path.parent), str(save_path.name) + 'txt')
    tf.train.Saver().save(session, str(save_path)[:-3] + '.chkp')

    freeze_graph.freeze_graph(str(save_path) + 'txt',
                              None, False,
                              str(save_path)[:-3] + '.chkp',
                              output_names[0],
                              "save/restore_all",
                              "save/Const:0",
                              str(save_path)[:-3] + '_frozen.pb',
                              clear_devices, "")
