"""Own Layers"""

import tensorflow as tf
from tensorlayer.layers.core import Layer
import _my_logging as logging
from tensorflow.python.framework import ops
import copy


class LayersConfig:
    tf_dtype = tf.float32  # TensorFlow DType
    set_keep = {}  # A dictionary for holding tf.placeholders


try:  # For TF12 and later
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
except Exception:  # For TF11 and before
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.VARIABLES


class ScaleLayer(Layer):
    def __init__(
            self,
            prev_layer,
            init_scale=0.05,
            name='scale',
    ):
        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs

        logging.info("ScaleLayer  %s: init_scale: %f" % (self.name, init_scale))
        with tf.variable_scope(name):
            # scale = tf.get_variable(name='scale_factor', init, trainable=True, )
            scale = tf.get_variable("scale", shape=[1], initializer=tf.constant_initializer(value=init_scale),
                                    trainable=False)
            self.outputs = self.inputs * scale

        self.all_layers.append(self.outputs)
        self.all_params.append(scale)


class Modify_BatchNormLayer(Layer):
    """
    The :class:`BatchNormLayer` is a batch normalization layer for both fully-connected and convolution outputs.
    See ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Parameters
    ----------
    layer : :class:`Layer`
            The previous layer.
    decay : float
            A decay factor for `ExponentialMovingAverage`.
            Suggest to use a large value for large dataset.
    epsilon : float
            Eplison.
    act : activation function
            The activation function of this layer.
    is_train : boolean
            Is being used for training or inference.
    beta_init : initializer or None
            The initializer for initializing beta, if None, skip beta.
            Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None
            The initializer for initializing gamma, if None, skip gamma.
            When the batch normalization layer is use instead of 'biases', or the next layer is linear, this can be
            disabled since the scaling can be done by the next layer. see `Inception-ResNet-v2 <https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py>`__
    dtype : TensorFlow dtype
            tf.float32 (default) or tf.float16.
    name : str
            A unique layer name.

    """

    def __init__(
            self,
            prev_layer,
            decay=0.9,
            # de_decay=0.1,
            epsilon=0.00001,
            act=tf.identity,
            is_train=False,
            is_act=1.0,  # work in the experiment of turning bn up or down
            # alpha_init=1.0,
            beta_init=tf.zeros_initializer,
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
            name='modify_batchnorm_layer',
    ):
        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.epsilon = epsilon
        self.inputs = prev_layer.outputs
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages

        with tf.variable_scope(name):
            axis = list(range(len(x_shape) - 1))
            # 1. beta, gamma
            variables = []
            if beta_init:
                if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                    beta_init = beta_init()
                beta = tf.get_variable('beta', shape=params_shape, initializer=beta_init, dtype=LayersConfig.tf_dtype,
                                       trainable=is_train)
                variables.append(beta)
            else:
                beta = None

            if gamma_init:
                gamma = tf.get_variable(
                    'gamma',
                    shape=params_shape,
                    initializer=gamma_init,
                    dtype=LayersConfig.tf_dtype,
                    trainable=is_train,
                )
                variables.append(gamma)
            else:
                gamma = None

            # 2.
            if tf.__version__ > '0.12.1':
                moving_mean_init = tf.zeros_initializer()
            else:
                moving_mean_init = tf.zeros_initializer
            moving_mean = tf.get_variable('moving_mean', params_shape, initializer=moving_mean_init,
                                          dtype=LayersConfig.tf_dtype, trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance',
                params_shape,
                initializer=tf.constant_initializer(1.),
                dtype=LayersConfig.tf_dtype,
                trainable=False
            )

            # 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            self.mean = mean
            self.variance = variance
            try:  # TF12
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay,
                                                                           zero_debias=False)  # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay, zero_debias=False)  # if zero_debias=True, has bias
            # logging.info("TF12 moving")
            except Exception:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)

            # logging.info("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                avg, var = mean_var_with_update()
                avg_ = tf.add(tf.multiply(is_act, avg), tf.multiply((1.0 - is_act), moving_mean))
                var_ = tf.add(tf.multiply(is_act, var), tf.multiply((1.0 - is_act), moving_variance))
                self.outputs = act(tf.nn.batch_normalization(self.inputs, avg_, var_, beta, gamma, epsilon))
            else:
                self.outputs = act(
                    tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon))

        self.all_layers.append(self.outputs)
        self.all_params.extend(variables)

    def get_value(self):
        return self.mean, tf.sqrt(self.variance + self.epsilon)


class GroupNormLayer1D(Layer):
    """
    The :class:`BatchNormLayer` is a batch normalization layer for both fully-connected and convolution outputs.
    See ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Parameters
    ----------
    layer : :class:`Layer`
            The previous layer.
    decay : float
            A decay factor for `ExponentialMovingAverage`.
            Suggest to use a large value for large dataset.
    epsilon : float
            Eplison.
    act : activation function
            The activation function of this layer.
    is_train : boolean
            Is being used for training or inference.
    beta_init : initializer or None
            The initializer for initializing beta, if None, skip beta.
            Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None
            The initializer for initializing gamma, if None, skip gamma.
            When the batch normalization layer is use instead of 'biases', or the next layer is linear, this can be
            disabled since the scaling can be done by the next layer. see `Inception-ResNet-v2 <https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py>`__
    dtype : TensorFlow dtype
            tf.float32 (default) or tf.float16.
    name : str
            A unique layer name.

    """

    def __init__(
            self,
            prev_layer,
            epsilon=0.001,
            act=tf.identity,
            group_num=16,
            is_train=True,
            # alpha_init=1.0,
            beta_init=tf.zeros_initializer,
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
            name='modify_batchnorm_layer',
    ):
        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]
        self.inputs = tf.reshape(self.inputs, [-1, -1, x_shape[2].value // group_num, group_num])
        self.epsilon = epsilon

        with tf.variable_scope(name):
            axis = list(range(1, (len(x_shape) - 1)))  # shape:[x, o, o, x]
            # 1. beta, gamma
            variables = []
            if beta_init:
                if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                    beta_init = beta_init()
                beta = tf.get_variable('beta', shape=params_shape, initializer=beta_init, dtype=LayersConfig.tf_dtype,
                                       trainable=is_train)
                variables.append(beta)
            else:
                beta = None

            if gamma_init:
                gamma = tf.get_variable(
                    'gamma',
                    shape=params_shape,
                    initializer=gamma_init,
                    dtype=LayersConfig.tf_dtype,
                    trainable=is_train,
                )
                variables.append(gamma)
            else:
                gamma = None

            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            self.mean = mean
            self.variance = variance

            with ops.name_scope(name, "groupnorm", [self.inputs, mean, variance, gamma, beta]):
                x = (self.inputs - mean) / tf.sqrt(variance + epsilon)
                x = tf.reshape(x, [-1, -1, x_shape[2].value])
                self.outputs = act(x * gamma + beta)

        self.all_layers.append(self.outputs)
        self.all_params.extend(variables)

    def get_value(self):
        return self.mean, tf.sqrt(self.variance + self.epsilon)


class OriginalBatchNormLayer(Layer):

    def __init__(
            self,
            prev_layer,
            decay=0.9,
            epsilon=0.00001,
            act=tf.identity,
            is_train=False,
            beta_init=tf.zeros_initializer,
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
            name='batchnorm_layer',
    ):

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        logging.info(
            "BatchNormLayer %s: decay:%f epsilon:%f act:%s is_train:%s" % (
            self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages

        with tf.variable_scope(name):
            axis = list(range(len(x_shape) - 1))
            # 1. beta, gamma
            variables = []
            if beta_init:
                if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                    beta_init = beta_init()
                beta = tf.get_variable('beta', shape=params_shape, initializer=beta_init, dtype=LayersConfig.tf_dtype,
                                       trainable=is_train)
                variables.append(beta)
            else:
                beta = None

            if gamma_init:
                gamma = tf.get_variable(
                    'gamma',
                    shape=params_shape,
                    initializer=gamma_init,
                    dtype=LayersConfig.tf_dtype,
                    trainable=is_train,
                )
                variables.append(gamma)
            else:
                gamma = None

            # 2.
            if tf.__version__ > '0.12.1':
                moving_mean_init = tf.zeros_initializer()
            else:
                moving_mean_init = tf.zeros_initializer
            moving_mean = tf.get_variable('moving_mean', params_shape, initializer=moving_mean_init,
                                          dtype=LayersConfig.tf_dtype, trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance',
                params_shape,
                initializer=tf.constant_initializer(1.),
                dtype=LayersConfig.tf_dtype,
                trainable=False,
            )

            # 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            self.mean = mean
            self.variance = variance
            try:  # TF12
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay,
                                                                           zero_debias=False)  # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay, zero_debias=False)  # if zero_debias=True, has bias
            # logging.info("TF12 moving")
            except Exception:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            # logging.info("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                mean, var = mean_var_with_update()
                self.outputs = act(tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon))
            else:
                self.outputs = act(
                    tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon))

            variables.extend([moving_mean, moving_variance])

        self.all_layers.append(self.outputs)
        self.all_params.extend(variables)

    def get_value(self):
        return self.mean, self.variance


def global_avg_net(net, name=None):
    scope_name = tf.get_variable_scope().name
    if scope_name:
        name = scope_name + '/' + name
    shape = net.outputs.get_shape()
    logging.info("GlobalAvgPool {:}: shape:{:}".format(name, shape))

    outputs = tf.reduce_mean(net.outputs, 1)
    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])

    return net_new


def global_max_net(net, name=None):
    scope_name = tf.get_variable_scope().name
    if scope_name:
        name = scope_name + '/' + name
    shape = net.outputs.get_shape()
    logging.info("GlobalMaxPool {:}: shape:{:}".format(name, shape))

    outputs = tf.reduce_max(net.outputs, 1)
    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])

    return net_new


def fw(x):
    with tf.get_default_graph().gradient_override_map({"Sign": "Identity"}):
        return tf.sign(x)


def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.) / 2., 0, 1)


def round(x):
    with tf.get_default_graph().gradient_override_map({"Round": "Identity"}):
        return tf.round(x)


def binary_value(x):
    ab = hard_sigmoid(x)
    ab = round(ab)  # 0.5 transforms to 0
    ab = ab * 2 - 1
    return ab


def binary_outputs(net, name=None):
    scope_name = tf.get_variable_scope().name
    if scope_name:
        name = scope_name + '/' + name
    shape = net.outputs.get_shape()
    logging.info("BinaryOutputs {:}: shape:{:}".format(name, shape))

    outputs = binary_value(net.outputs)
    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])

    return net_new
