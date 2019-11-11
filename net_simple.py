from __future__ import print_function

# internal import
import tensorflow as tf
import tensorlayer as tl
import own_layers


def network_structure(inputs, is_training=False, **kwargs):
    with tf.variable_scope('full_net'):
        tl.layers.set_name_reuse(True)

        # input
        network = tl.layers.InputLayer(inputs=inputs, name='input_layer')

        # CNN
        conv_layer_num = 9
        filter_width = 11
        keep = 0.5
        for l in range(conv_layer_num):
            cnn_name = 'cnn_' + str(l)
            p_name = 'avgpool_' + str(l)
            drop_name = 'dropout_' + str(l)
            # bn_name = 'bnorm_' + str(l)
            ln_name = 'lnorm_' + str(l)

            in_channels = network.outputs.get_shape().as_list()[2]
            if l == 0:
                out_channels = 64
            elif l % 2 == 0:
                out_channels = in_channels + 46
            else:
                out_channels = in_channels

            network = tl.layers.Conv1dLayer(network,
                                            shape=(filter_width, in_channels, out_channels),
                                            padding='SAME',
                                            name=cnn_name)

            network = tl.layers.LayerNormLayer(network,
                                               act=tf.nn.relu,
                                               trainable=is_training,
                                               name=ln_name)

            network = tl.layers.MeanPool1d(network, 2, 2, padding='SAME', name=p_name)
            if l != conv_layer_num - 1:
                # except last layer
                network = tl.layers.DropoutLayer(network, keep=keep, is_fix=True,
                                                 is_train=is_training, name=drop_name)

        # Global Average Pooling
        network = own_layers.global_avg_net(network, name='global_avg_pool')

        network = tl.layers.DropoutLayer(network, keep=keep, is_fix=True,
                                         is_train=is_training, name=drop_name)

        # Output
        network = tl.layers.DenseLayer(network, n_units=4, name='output_layer')

        train_params = network.all_params

    return network, network.outputs, train_params
