""" Calculate gradients """

import tensorflow as tf
from get_variables import get_variables_with_name


def get_gradients(loss, layer_num, variable_info):

	grads = [None for _ in range(layer_num)]
	grads_sum = None
	for layer in range(layer_num):
		if layer == (layer_num - 1):
			addition_info = 'output'
			variable_info[-1] = addition_info
		elif layer == 0:
			addition_info = 'cnn_' + str(layer) + '/'
			variable_info.append(addition_info)
		else:
			addition_info = 'cnn_' + str(layer) + '/'
			variable_info[-1] = addition_info
		variables = get_variables_with_name(variable_name=variable_info, name=addition_info)
		grads_temp = tf.gradients(loss, variables)
		grads[layer] = tf.norm(grads_temp, ord=2)
		if grads_sum is None:
			grads_sum = tf.reduce_sum(tf.square(grads_temp))
		else:
			grads_sum += tf.reduce_sum(tf.square(grads_temp))
	grads_all = tf.sqrt(grads_sum + 0.00001)

	return grads, grads_all
