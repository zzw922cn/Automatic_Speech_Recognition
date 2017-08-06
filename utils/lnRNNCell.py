#coding=utf-8
'''
Layer Normalization based on RNN
'''


import tensorflow as tf
import numpy as np

def ln(layer, gain, bias):
    self.dims = layer.get_shape().as_list()
    assert len(self.dims)==3, 'layer must be 3-D tensor'
    miu, sigma = tf.nn.moments(self.layer, axes=[2], keep_dims=True)
    ln_layer = tf.div(tf.subtract(self.layer, miu), tf.sqrt(sigma))
    ln_layer = ln_layer*gain + bias
    return ln_layer

class BasicRNNCell(tf.contrib.rnn.RNNCell):
  """The most basic RNN cell.
  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """

  def __init__(self, num_units, activation=None, reuse=None):
    super(BasicRNNCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or tf.nn.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    with tf.variable_scope('layer_normalization'):
      gain = tf.get_variable('gain', shape=[self._num_units], initializer=tf.ones_initializer())
      bias = tf.get_variable('bias', shape=[self._num_units], initializer=tf.zeros_initializer())
    output = ln(_linear([inputs, state], self._num_units, True), gain, bias)
    return output, output

class GRUCell(tf.contrib.rnn.RNNCell):

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(GRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or tf.nn.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope('layer_normalization'):
      gain1 = tf.get_variable('gain1', shape=[2*self._num_units], initializer=tf.ones_initializer())
      bias1 = tf.get_variable('bias1', shape=[2*self._num_units], initializer=tf.zeros_initializer())
      gain2 = tf.get_variable('gain2', shape=[self._num_units], initializer=tf.ones_initializer())
      bias2 = tf.get_variable('bias2', shape=[self._num_units], initializer=tf.zeros_initializer())

    with vs.variable_scope("gates"):  # Reset gate and update gate.
      # We start with bias of 1.0 to not reset and not update.
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        dtype = [a.dtype for a in [inputs, state]][0]
        bias_ones = tf.constant_initializer(1.0, dtype=dtype)
      value = tf.nn.sigmoid(ln(
          _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                  self._kernel_initializer), gain1, bias1))
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    with vs.variable_scope("candidate"):
      c = self._activation(ln(
          _linear([inputs, r * state], self._num_units, True,
                  self._bias_initializer, self._kernel_initializer), gain2, bias2))
    new_h = u * state + (1 - u) * c
    return new_h, new_h

class BasicLSTMCell(tf.contrib.rnn.RNNCell):

  def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None):
    super(BasicLSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    with tf.variable_scope('layer_normalization'):
      gain_h = tf.get_variable('gain_h', shape=[4*self._num_units], initializer=tf.ones_initializer())
      bias_h = tf.get_variable('bias_h', shape=[4*self._num_units], initializer=tf.zeros_initializer())
      gain_c = tf.get_variable('gain_c', shape=[self._num_units], initializer=tf.ones_initializer())
      bias_c = tf.get_variable('bias_c', shape=[self._num_units], initializer=tf.zeros_initializer())
    sigmoid = math_ops.sigmoid
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

    concat = ln(_linear([inputs, h], 4 * self._num_units, True), gain_h, bias_h)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

    new_c = (
        c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
    new_h = self._activation(ln(new_c, gain_c, bias_c)) * sigmoid(o)

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state
