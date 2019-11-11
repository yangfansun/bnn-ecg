from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _my_logging as logging

import tensorflow as tf
from tensorlayer.layers import Layer

tf.logging.set_verbosity(tf.logging.ERROR)


class Binarized_DenseLayer(Layer):
	def __init__(
			self,
			layer=None,
			n_units=100,
			act=tf.identity,
			binarized_weight=tf.identity,
			W_init=tf.truncated_normal_initializer(stddev=0.1),
			b_init=tf.constant_initializer(value=0.0),
			W_init_args={},
			b_init_args={},
			name='b_dense_layer',
	):
		Layer.__init__(self, name=name)
		self.inputs = layer.outputs
		if self.inputs.get_shape().ndims != 2:
			raise Exception("The input dimension must be rank 2, please reshape or flatten it")

		n_in = int(self.inputs.get_shape()[-1])
		self.n_units = n_units
		logging.info("Binarized_DenseLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
		with tf.variable_scope(name) as vs:
			W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, **W_init_args)
			bin_w = binarized_weight(W)
			if b_init is not None:
				try:
					b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, **b_init_args)
				except:  # If initializer is a constant, do not specify shape.
					b = tf.get_variable(name='b', initializer=b_init, **b_init_args)
				self.outputs = act(tf.matmul(self.inputs, bin_w) + b)
			else:
				self.outputs = act(tf.matmul(self.inputs, bin_w))

		# Hint : list(), dict() is pass by value (shallow), without them, it is
		# pass by reference.
		self.all_layers = list(layer.all_layers)
		self.all_params = list(layer.all_params)
		self.all_drop = dict(layer.all_drop)
		self.all_layers.extend([self.outputs])
		if b_init is not None:
			self.all_params.extend([W, b])
		else:
			self.all_params.extend([W])


class Binarized_Convolution1D(Layer):
	def __init__(
			self,
			layer=None,
			shape=(5, 1, 5),
			stride=1,
			act=tf.identity,
			dilation_rate=1,
			padding='SAME',
			binarize_weight=tf.identity,
			W_init=tf.truncated_normal_initializer(stddev=0.02),
			b_init=None,
			W_init_args={},
			b_init_args={},
			use_cudnn_on_gpu=True,
			data_format='NWC',
			name='cnn_layer',
	):
		Layer.__init__(self, name=name)
		self.inputs = layer.outputs
		logging.info("BinarizedConvolution1D %s: shape:%s strides:%s pad:%s activation:%s" %
					(self.name, str(shape), str(stride), padding, act.__name__))

		with tf.variable_scope(name) as vs:
			W = tf.get_variable(name='W_conv1d', shape=shape, initializer=W_init, **W_init_args)
			bin_w = binarize_weight(W)
			self.outputs = tf.nn.convolution(
				self.inputs,
				bin_w,
				strides=(stride,),
				padding=padding,
				dilation_rate=(dilation_rate,),
				data_format=data_format
			)
			if b_init:
				b = tf.get_variable(name='b_conv1d', shape=(shape[-1]), initializer=b_init, **b_init_args)
				self.outputs = self.outputs + b

			self.outputs = act(self.outputs)

		self.all_layers = list(layer.all_layers)
		self.all_params = list(layer.all_params)
		self.all_drop = dict(layer.all_drop)
		self.all_layers.extend([self.outputs])
		if b_init:
			self.all_params.extend([W, b])
		else:
			self.all_params.extend([W])


class Binarized_Outputs(Layer):
	def __init__(
			self,
			layer=None,
			act=tf.identity,
			name='b_out'
	):
		Layer.__init__(self, name=name)
		self.outputs = layer.outputs
		self.outputs = act(self.outputs)

		self.all_layers = list(layer.all_layers)
		self.all_params = list(layer.all_params)
		self.all_drop = dict(layer.all_drop)
		self.all_layers.extend([self.outputs])
