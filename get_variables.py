"""Get variables with name"""

import tensorflow as tf


def get_variables_with_name(variable_name, delete=None, train_only=True, name=None, if_print=True):
	"""
	get network parameters with it's name
	:param train_only:
	:param variable_name:
	:param name: variables' name, a list or a tuple
	:param delete: delete the variables including these words
	:param name:
	:param if_print:
	:return:
	"""

	out_vars = []
	if train_only:
		in_vars = tf.trainable_variables()
	else:
		in_vars = tf.global_variables()
	d_vars = [var for var in in_vars]
	for temp in range(len(variable_name)):
		for var in d_vars:
			if variable_name[temp] in var.name:
				if delete:
					if delete in var.name:
						continue
				out_vars.append(var)
		d_vars = out_vars
		out_vars = []
	if if_print:
		if name:
			print(name)
		for idx, v in enumerate(d_vars):
			print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

	return d_vars
