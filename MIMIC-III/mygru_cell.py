# -*- coding: utf-8 -*-
"""
Created on 2021/06/26

@author: JiaWei OuYang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn import RNNCell
import tensorflow as tf
if "1.12" in tf.__version__ or "1.15" in tf.__version__:
    from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
    from tensorflow.python.layers import base as base_layer
    from tensorflow.python.ops import nn_ops
    _BIAS_VARIABLE_NAME = "bias"
    _WEIGHTS_VARIABLE_NAME = "kernel"
    
    class MyGRUCell15(LayerRNNCell):
      """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    
      Args:
        num_units: int, The number of units in the GRU cell.
        activation: Nonlinearity to use.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.  If not `True`, and the existing scope already has
         the given variables, an error is raised.
        kernel_initializer: (optional) The initializer to use for the weight and
        projection matrices.
        bias_initializer: (optional) The initializer to use for the bias.
        name: String, the name of the layer. Layers with the same name will
          share weights, but to avoid mistakes we require reuse=True in such
          cases.
      """
      def __init__(self,
                   num_units,
                   size_inputs,
                   activation=None,
                   reuse=None,
                   kernel_initializer=None,
                   bias_initializer=None,
                   name=None):
        super(MyGRUCell15, self).__init__(_reuse=reuse, name=name)
    
        self._size_inputs = size_inputs
        self._num_units = num_units
        self._num_units_half = int(num_units/2)
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
    
      @property
      def state_size(self):
        return self._num_units
    
      @property
      def output_size(self):
        return self._size_inputs
    
      def build(self, inputs_shape):
        if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        if inputs_shape[1].value != self._size_inputs * 2 + self._num_units_half:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)


        self.input_depth = self._size_inputs
        self.diagnoal_constant = tf.constant(1., shape=[self.input_depth, self.input_depth]) - tf.matrix_diag(tf.constant(1., shape=[self.input_depth, ]))

        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self.input_depth + self._num_units_half, 2 * self._num_units_half],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units_half],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self.input_depth + self._num_units_half, self._num_units_half],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units_half],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._even_kernel = self.add_variable(
            "even/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self._num_units, self._num_units_half],
            initializer=self._kernel_initializer)
        self._even_bias = self.add_variable(
            "even/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units_half],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))

        self._even_gated_kernel = self.add_variable(
            "even_gated/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self._num_units_half, self._num_units_half],
            initializer=self._kernel_initializer)
        self._even_gated_bias = self.add_variable(
            "even_gated/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units_half],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))

# =================================================================

        self._input_kernel = self.add_variable(
            "input/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self._num_units_half, self.input_depth],
            initializer=self._kernel_initializer)
        self._input_bias = self.add_variable(
            "input/%s" % _BIAS_VARIABLE_NAME,
            shape=[self.input_depth],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))

        self._feature_kernel = self.add_variable(
            "feature/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self.input_depth, self.input_depth],
            initializer=self._kernel_initializer)
        self._feature_bias = self.add_variable(
            "feature/%s" % _BIAS_VARIABLE_NAME,
            shape=[self.input_depth],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))

        self._input_beta_kernel = self.add_variable(
            "input_beta/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self.input_depth, self.input_depth],
            initializer=self._kernel_initializer)
        self._input_beta_bias = self.add_variable(
            "input_beta/%s" % _BIAS_VARIABLE_NAME,
            shape=[self.input_depth],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))

# =================================================================
        self._gate_kernel_m = self.add_variable(
            "gates_m/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self.input_depth + self._num_units_half, 2 * self._num_units_half],
            initializer=self._kernel_initializer)
        self._gate_bias_m = self.add_variable(
            "gates_m/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units_half],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel_m = self.add_variable(
            "candidate_m/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self.input_depth + self._num_units_half, self._num_units_half],
            initializer=self._kernel_initializer)
        self._candidate_bias_m = self.add_variable(
            "candidate_m/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units_half],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._even_kernel_m = self.add_variable(
            "even_m/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[self._num_units, self._num_units_half],
            initializer=self._kernel_initializer)
        self._even_bias_m = self.add_variable(
            "even_m/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units_half],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))

        self.built = True
    
      def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        totalLength=inputs.get_shape().as_list()[1]
        x_input = inputs[:, 0:self.input_depth]
        # m_input = inputs[:, self.input_depth:totalLength]
        m_input = inputs[:, self.input_depth:(totalLength-self._num_units_half)]
        rth=inputs[:, totalLength-self._num_units_half:]

        x_state = state[:, 0:self._num_units_half]
        m_state = state[:, self._num_units_half:self._num_units]

        x_state=math_ops.multiply(rth,x_state)
#==============================================
        # Additive combination
        # even_inputs = math_ops.matmul(
        #     array_ops.concat([x_state, m_state], 1), self._even_kernel)
        # even_inputs = nn_ops.bias_add(even_inputs, self._even_bias)

        # Gated combination
        even_inputs = math_ops.matmul(
            math_ops.multiply(math_ops.sigmoid(m_state),x_state), self._even_gated_kernel)
        even_inputs = nn_ops.bias_add(even_inputs, self._even_gated_bias)

        even_inputs = self._activation(even_inputs)
#==============================================

#==============================================
        o_input = math_ops.matmul(even_inputs, self._input_kernel)
        o_input = nn_ops.bias_add(o_input, self._input_bias)

        feature_input = math_ops.matmul(x_input, tf.multiply(self._feature_kernel, self.diagnoal_constant))
        feature_input = nn_ops.bias_add(feature_input, self._feature_bias)


        beta_weight = math_ops.matmul(m_input, self._input_beta_kernel)
        beta_weight = nn_ops.bias_add(beta_weight, self._input_beta_bias)
        beta_weight = math_ops.sigmoid(beta_weight)


        o_input = math_ops.multiply((beta_weight),feature_input)+math_ops.multiply((1-beta_weight),o_input)
        
        x_input = math_ops.multiply(m_input,x_input)+math_ops.multiply((1-m_input),o_input)

#==============================================
        gate_inputs = math_ops.matmul(
            array_ops.concat([x_input, even_inputs], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
    
        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    
        r_state = r * x_state
    
        candidate = math_ops.matmul(
            array_ops.concat([x_input, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)
    
        c = self._activation(candidate)
        x_state = u * x_state + (1 - u) * c

      #  m矩阵的处理

        # even_inputs_m = math_ops.matmul(
        #     array_ops.concat([x_state, m_state], 1), self._even_kernel_m)
        # even_inputs_m = nn_ops.bias_add(even_inputs_m, self._even_bias_m)

        even_inputs_m = m_state
        
        gate_inputs_m = math_ops.matmul(
            array_ops.concat([m_input, even_inputs_m], 1), self._gate_kernel_m)
        gate_inputs_m = nn_ops.bias_add(gate_inputs_m, self._gate_bias_m)
    
        value_m = math_ops.sigmoid(gate_inputs_m)
        r_m, u_m = array_ops.split(value=value_m, num_or_size_splits=2, axis=1)
    
        r_state_m = r_m * m_state
    
        candidate_m = math_ops.matmul(
            array_ops.concat([m_input, r_state_m], 1), self._candidate_kernel_m)
        candidate_m = nn_ops.bias_add(candidate_m, self._candidate_bias_m)
    
        c_m = self._activation(candidate_m)
        m_state = u_m * m_state + (1 - u_m) * c_m
        # next state
        new_h = array_ops.concat([x_state, m_state], 1)

        return o_input,new_h