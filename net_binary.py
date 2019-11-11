from __future__ import print_function

import _my_logging as logging
import tensorflow as tf
import tensorlayer as tl
import binary_layers
import own_layers

tf.logging.set_verbosity(tf.logging.ERROR)


def network_structure(inputs, is_training=False, **kwargs):
    with tf.variable_scope('binary_net'):
        tl.layers.set_name_reuse(True)

        scale = 1.0
        keep = 0.5
        conv_layer_num = 9

        # input
        network = tl.layers.InputLayer(inputs=inputs, name='b_input_layer')
        # network = tl.layers.DropoutLayer(layer=network, keep=0.8, is_fix=True,
        #                                  is_train=is_training, name='dropout_0')

        for l in range(conv_layer_num):
            cnn_name = 'cnn_' + str(l)
            # mp_name = 'maxpool_' + str(l)
            ap_name = 'avgpool_' + str(l)
            drop_name = 'dropout_' + str(l)
            bn_name = 'bnorm_' + str(l)
            # ln_name = 'lnorm_' + str(l)

            # channel choice
            filter_width = 11
            # filter_width = 7
            # filter_width = 9
            in_channels = network.outputs.get_shape().as_list()[2]
            if l == 0:
                out_channels = 64
            elif l % 2 == 0:
                out_channels = in_channels + 46
            else:
                out_channels = in_channels

            # weight binary choice
            if l == 0:
                weight = tf.identity
            else:
                weight = own_layers.binary_value

            if l == 0:
                input = network.outputs
                logging.debug('input %s' % input)
            network = binary_layers.Binarized_Convolution1D(network,
                                                            act=tf.identity,
                                                            binarize_weight=weight,
                                                            shape=(filter_width, in_channels, out_channels),
                                                            name=cnn_name)

            network = tl.layers.MeanPool1d(network, 2, 2, padding='SAME', name=ap_name)
            # network = tl.layers.MaxPool1d(network, 2, 2, padding='SAME', name=mp_name)

            if l == (conv_layer_num - 1):
                network = own_layers.Modify_BatchNormLayer(network,
                                                           decay=kwargs['decay'],
                                                           act=tf.identity,
                                                           is_train=is_training,
                                                           is_act=kwargs['is_act'],
                                                           name=bn_name)
            else:
                network = own_layers.Modify_BatchNormLayer(network,
                                                           decay=kwargs['decay'],
                                                           act=own_layers.binary_value,
                                                           is_train=is_training,
                                                           is_act=kwargs['is_act'],
                                                           name=bn_name)
                # Get mean and variance in the batch norm at layer-1
                if l == 0:
                    mean, var = network.get_value()
                network = tl.layers.DropoutLayer(network, keep=keep, is_fix=True,
                                                 is_train=is_training, name=drop_name)

        network = own_layers.global_avg_net(network, name='b_global_avg_pool')
        network = own_layers.binary_outputs(network, name='b_binary_outputs')
        network = tl.layers.DropoutLayer(network, keep=keep, is_fix=True,
                                         is_train=is_training, name='b_dropout_out')

        # output
        network = binary_layers.Binarized_DenseLayer(layer=network,  # decrease the numbers of dense layer
                                                     n_units=4,
                                                     name='b_dense_output')
        network = own_layers.ScaleLayer(network, init_scale=scale, name='b_scale_layer')

        train_params = network.all_params
        layer_num = conv_layer_num + 1

    return network, network.outputs, train_params, layer_num, [mean, var]
