# coding: utf-8
import os
import onnx
from onnx_tf.backend import prepare
import keras2onnx
import tf2onnx
from tf2onnx import utils, optimizer, tf_loader
from tensorflow.core.framework import graph_pb2
import tensorflow as tf

def convert(input_ops_dict, output_ops, input_model, output_model):
    '''Convert keras h5 to tensorflow pb
        Args:
            input_ops_dict: input ops dict including names and shapes
            output_ops: output op names
            input_model: input keras h5 model name
            output_model: output pb model name
    '''
    onnx_name = ".tmp.onnx"
    pb_name = ".tmp.pb"
    # keras --> onnx --> pb --> onnx --> pb

    # keras --> onnx
    model = tf.keras.models.load_model(input_model)
    onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=8)
    keras2onnx.save_model(onnx_model, onnx_name)

    # onnx --> tf
    onnx_model = onnx.load(onnx_name)
    tf_rep = prepare(onnx_model, input_shape_dict=input_ops_dict)
    tf_rep.export_graph(pb_name)

    # tf --> onnx (fold constants)
    inputs = input_ops_dict.keys()
    inputs = [i + ":0" for i in inputs]
    outputs = output_ops
    graph_def, inputs, outputs = tf_loader.from_graphdef(pb_name, inputs, outputs)
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name="")

    g = tf2onnx.tfonnx.process_tf_graph(tf_graph, opset=8,
                input_names=inputs,
                output_names=outputs)
    onnx_graph = optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model("converted from %s" % pb_name)
    utils.save_protobuf(onnx_name, model_proto)

    # onnx --> tf
    onnx_model = onnx.load(onnx_name)
    tf_rep = prepare(onnx_model, input_shape_dict=input_ops_dict)
    tf_rep.export_graph(output_model)

    # remove tmp files
    if os.path.exists(onnx_name):
        os.remove(onnx_name)
    if os.path.exists(pb_name):
        os.remove(pb_name)

