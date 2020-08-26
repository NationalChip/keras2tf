from functools import partial

import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.python.ops import array_ops

from onnx_tf.common import exception

class RNNState(object):
  INPUT_INDEX = 0
  OUTPUT_INDEX = 0

  @classmethod
  def gen_lstm_input_state_names(cls):
      index = cls.INPUT_INDEX
      cls.INPUT_INDEX += 1
      return ("State_c%d" % index, "State_h%d" % index)

  @classmethod
  def gen_lstm_output_state_names(cls, increase):
      index = cls.OUTPUT_INDEX
      if increase:
        cls.OUTPUT_INDEX += 1
      return ("State_c%d_out" % index, "State_h%d_out" % index)

  @classmethod
  def get_lstm_out_state_names(cls):
    names = []
    for i in range(cls.OUTPUT_INDEX):
      names.append("State_c%d_out" % i)
      names.append("State_h%d_out" % i)
    return names

  @classmethod
  def reset(cls):
    cls.INPUT_INDEX = 0
    cls.OUTPUT_INDEX = 0

class RNNMixin(object):

  ONNX_ACTIVATION_MAPPING = {
      # Added from tf 1.8
      # "affine": tf.contrib.distributions.bijectors.AffineScalar,
      # tf.contrib was removed since tf 2.0,
      # Class Affine had been move to the following module
      # "affine": tfp.bijectors.Affine,
      "elu": tf.nn.elu,
      "hard_sigmoid": tf.keras.backend.hard_sigmoid,
      "leaky_relu": tf.nn.leaky_relu,
      "relu": tf.nn.relu,
      "sigmoid": tf.sigmoid,
      "softsign": tf.nn.softsign,
      "softplus": tf.nn.softplus,
      "tanh": tf.tanh,
      "thresholded_relu": tf.keras.layers.ThresholdedReLU,
  }

  @classmethod
  def rnn(cls, x, cell_class, cell_kwargs, rnn_kwargs, activations, direction):
    def gx_rnn(cell_fw, x):
      outputs=[]
      #state_c_name, state_h_name = RNNState.gen_lstm_input_state_names()
      #batch_size = 1
      #state_c = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_kwargs["num_units"]], name=state_c_name)
      #state_h = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_kwargs["num_units"]], name=state_h_name)
      state_c, state_h = rnn_kwargs["state_c"], rnn_kwargs["state_h"]
      states = (tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_c, state_h),)
      for step in range(1):
        cell_output, states = cell_fw(x[0], states)
        state_c_out = tf.compat.v1.identity(states[0][0])
        state_h_out = tf.compat.v1.identity(states[0][1])
        outputs.append(cell_output)
      out_states = (tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_c_out, state_h_out),)
      return outputs, out_states

    cell_kwargs["activation"] = activations[0]

    rnn_cell = [cell_class(**cell_kwargs)]
    cell_fw = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_cell, state_is_tuple=True)

    if direction == "bidirectional":
      cell_kwargs["activation"] = activations[1]
      rnn_cell_bw = [cell_class(**cell_kwargs)]
      cell_bw = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_cell_bw)

    if direction == "forward":
      #states = cell_fw.zero_state(batch_size, tf.float32)
      outputs, states = gx_rnn(cell_fw, x)
      #outputs, states = tf.compat.v1.nn.dynamic_rnn(cell_fw, x, **rnn_kwargs)
      rnn_outputs = tf.stack(outputs)
    elif direction == "bidirectional":
      outputs, states = tf.compat.v1.nn.bidirectional_dynamic_rnn(
          cell_fw, cell_bw, x, **rnn_kwargs)
    elif direction == "reverse":

      def _reverse(input_, seq_dim):
        return array_ops.reverse(input_, axis=[seq_dim])

      time_dim = 0
      inputs_reverse = _reverse(x, time_dim)
      outputs, states = tf.compat.v1.nn.dynamic_rnn(cell_fw, inputs_reverse,
                                                    **rnn_kwargs)
      outputs = _reverse(outputs, time_dim)

    return outputs, states

  @classmethod
  def rnn_get_activation(cls, name, alpha, beta):
    if name not in cls.ONNX_ACTIVATION_MAPPING:
      exception.OP_UNSUPPORTED_EXCEPT("Activation function {} for {}".format(
          name, cls.__name__), "Tensorflow")
    activation = cls.ONNX_ACTIVATION_MAPPING[name]
    kwargs = {}
    if name == "affine":
      kwargs["scale"] = alpha
      kwargs["shift"] = beta
      activation = activation(**kwargs)
    elif name == "elu":
      if alpha != 1:
        exception.OP_UNSUPPORTED_EXCEPT(
            "Activation function {} with alpha={} for {}".format(
                name, alpha, cls.__name__), "Tensorflow")
    elif name == "hard_sigmoid":
      if alpha != 0.2 or beta != 0.5:
        exception.OP_UNSUPPORTED_EXCEPT(
            "Activation function {} with alpha={}, beta={} for {}".format(
                name, alpha, beta, cls.__name__), "Tensorflow")
    elif name == "leaky_relu":
      kwargs["alpha"] = alpha or 0.01
      activation = partial(activation, **kwargs)
    elif name == "thresholded_relu":
      kwargs["theta"] = alpha
      activation = activation(**kwargs)
    return activation
