"""set the model and train"""
# -*- coding: utf8 -*-
from __future__ import print_function

import time
import os
# internal import
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import net_res
import net_simple
import net_binary
import logging
from gradients_statistic import get_gradients
from get_variables import get_variables_with_name

# import train_latest

TOTAL_NUM = 8528
SAMPLE_NUM = 8228
SEQUENCE_LENGTH = [6000, 9100, 12000, 16000, 18300]

NET_LIST = {'simplenet': net_simple,
            'binarynet': net_binary}

net_struct = NET_LIST['simplenet']

logger = logging.getLogger('')
logger.setLevel(logging.INFO)


def read_record(path):
    # read the length record
    load_size, train_length, val_length, pad_length = 0, 0, 0, 0
    record_iterator = tf.python_io.tf_record_iterator(path)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        train_length = (example.features.feature['train_length'].bytes_list.value[0])
        train_length = np.fromstring(train_length, dtype=np.int64)
        pad_length = (example.features.feature['pad_length'].bytes_list.value[0])
        pad_length = np.fromstring(pad_length, dtype=np.int64)

    return {'train_length': train_length, 'val_length': val_length, 'pad_length': pad_length}


def read_data(path, batch_size, is_train=False, length_arr=None, length_index=None):
    def example_parser(serialized_example):
        # read the training data

        features = tf.parse_single_example(serialized_example,
                                           features={'length': tf.FixedLenFeature([], tf.int64),
                                                     'input': tf.FixedLenFeature([], tf.string),
                                                     'label': tf.FixedLenFeature([], tf.int64)})

        input = tf.decode_raw(features['input'], out_type=tf.float32)
        size = tf.cast(features['length'], tf.int32)
        input = tf.reshape(input, [size, 1])
        label = features['label']

        # augmentation part
        if is_train:
            load_size = length_arr[length_index]
            input = augmentation(input, load_size=int(load_size))

        return input, label

    filename = [path]
    dataset = tf.contrib.data.TFRecordDataset(filename)
    dataset = dataset.map(example_parser)
    if is_train:
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=300)
        iterator = dataset.make_initializable_iterator()
    else:
        dataset = dataset.batch(1)
        iterator = dataset.make_initializable_iterator()
    # dataset = dataset.repeat(1)

    inputs, labels = iterator.get_next()

    return inputs, labels, iterator


def f1_measure(logits, labels):
    tp = []
    label_sum = []
    logit_sum = []
    for type_num in range(4):
        # logits = np.argmax(outputs, 1)
        index_label = np.where(labels == type_num)
        index_logit = np.where(logits == type_num)
        temp_labels = np.shape(index_logit)[1]
        temp_logits = np.shape(index_label)[1]
        temp = np.equal(logits[index_label], labels[index_label])
        tp_ = np.shape(np.where(temp))[1]  # Find out the true index
        label_sum.append(temp_labels)
        logit_sum.append(temp_logits)
        tp.append(tp_)
    label_sum = np.array(label_sum, dtype=np.float32)
    logit_sum = np.array(logit_sum, dtype=np.float32)
    tp = np.array(tp, dtype=np.float32)
    f1 = 2 * tp / (label_sum + logit_sum)

    return np.mean(f1[0:3]), f1


def augmentation(inputs, load_size=720, is_training=True):
    # inputs = tl.prepro.crop_multi(tf.reshape(inputs, [-1, 1, load_size, 2]), 1, int(load_size * 0.9),
    #                               is_random=is_training)

    if is_training:
        inputs = tf.random_crop(tf.reshape(inputs, [1, load_size, 1]), [1, int(0.9 * load_size), 1])
        inputs = inputs * tf.random_uniform([1], 0.9, 1.1, dtype=tf.float32)
    # else:
    # 	inputs = tf.image.central_crop(tf.reshape(inputs, [1, load_size, 1]), 0.9)
    # inputs = tf.image.resize_image_with_crop_or_pad(inputs, 1, load_size)
    # inputs = tf.image.resize_images(inputs, [1, load_size, 1])
    inputs = tf.reshape(inputs, [int(load_size * 0.9), 1])

    return inputs


def loss_calculate(outputs, labels, regularization, is_train=False, use_l2=True, clip=True, name=None):
    # calculate loss and accuracy

    # L2 for the MLP
    l2 = 0
    if is_train:
        if clip:
            var_name = ['cnn_0/W_conv1d']
            for p in get_variables_with_name(var_name, name='l2'):
                l2 += tf.contrib.layers.l2_regularizer(regularization)(p)
            var_name = ['dense', 'W']
            for p in get_variables_with_name(var_name):
                l2 += tf.contrib.layers.l2_regularizer(regularization)(p)
        else:
            var_name = ['W']
            for p in get_variables_with_name(var_name, name='l2'):
                l2 += tf.contrib.layers.l2_regularizer(regularization)(p)

    # Set loss and accuracy of training and validation respectively.
    prob = tf.clip_by_value(tf.nn.softmax(outputs), 1e-10, 1.0)
    labels_one = tf.one_hot(labels, 4, dtype=tf.float32)
    cross_entropy = -labels_one * tf.log(prob)
    loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, name=name, axis=1))
    if use_l2:
        loss += l2

    correct_prediction = tf.equal(tf.argmax(outputs, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

    return loss, accuracy, prob


def loss_calculate_dist(outputs, labels, outputs_prior=None, regularization=0.0, is_train=False, use_l2=True, clip=True,
                        prior_scale=1, train_scale=1, hard_label=False, name=None, **kwargs):
    # calculate loss and accuracy
    # L2 for the MLP
    l2 = 0
    if is_train:
        if clip:
            var_name = ['cnn_0/W_conv1d']
            for p in get_variables_with_name(var_name, name='l2'):
                l2 += tf.contrib.layers.l2_regularizer(regularization)(p)
            var_name = ['dense', 'W']
            for p in get_variables_with_name(var_name):
                l2 += tf.contrib.layers.l2_regularizer(regularization)(p)
        else:
            var_name = ['W']
            for p in get_variables_with_name(var_name, name='l2'):
                l2 += tf.contrib.layers.l2_regularizer(regularization)(p)

    # Set loss and accuracy of training and validation respectively.
    outputs_ = outputs * train_scale
    prob = tf.clip_by_value(tf.nn.softmax(outputs_), 1e-10, 1.0)
    if is_train:
        outputs_prior = outputs_prior * prior_scale
        outputs_prior = tf.cast(outputs_prior, dtype=tf.float32)
        outputs_prior_stop = tf.stop_gradient(outputs_prior)
        cross_entropy = -tf.nn.softmax(outputs_prior_stop) * tf.log(prob)
        loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, name=name, axis=1))
        if hard_label:
            # output_recover = tf.reciprocal(train_scale)
            # outputs_ = tf.multiply(outputs, output_recover)
            prob_ = tf.clip_by_value(tf.nn.softmax(outputs), 1e-10, 1.0)
            cross_entropy_ = -tf.one_hot(labels, 4, dtype=tf.float32) * tf.log(prob_)
            loss_ = tf.reduce_mean(tf.reduce_sum(cross_entropy_, axis=1))
            loss += kwargs['cut_value'] * loss_
        correct_prediction = tf.equal(tf.argmax(outputs, 1), labels)
    else:
        labels_one = tf.one_hot(labels, 4, dtype=tf.float32)
        cross_entropy = -labels_one * tf.log(prob)
        loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, name=name, axis=1))
        correct_prediction = tf.equal(tf.argmax(outputs, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

    if use_l2:
        loss += l2

    return loss, accuracy, prob


def get_op(outputs, labels, regularization, learning_rate, train_params, use_l2=True, clip=True):
    # Generate the training operation and the variables needed.

    loss, accuracy, _ = loss_calculate(outputs, labels, regularization, is_train=True, use_l2=use_l2, clip=clip,
                                       name='loss')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=train_params)

    return loss, accuracy, train_op


def get_op_dist(outputs, labels, outputs_prior, regularization, learning_rate, train_params, use_l2=True, clip=True,
                prior_scale=1, train_scale=1, hard_label=False, **kwargs):
    # Generate the training operation and the variables needed.

    loss, accuracy, _ = loss_calculate_dist(outputs, labels, outputs_prior, regularization, is_train=True,
                                            use_l2=use_l2,
                                            clip=clip, prior_scale=prior_scale, train_scale=train_scale,
                                            hard_label=hard_label,
                                            name='loss', cut_value=kwargs['hard_label_cut_value'])
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=train_params)

    return loss, accuracy, train_op


def generate_op(*args):
    return [args[i] for i in range(len(args))]


def category_accuracy(logits, labels):
    acc = []
    for type_num in range(4):
        index_label = np.where(labels == type_num)
        global_amount = np.shape(index_label)[1]
        temp = np.equal(logits[index_label], type_num)
        true_amount = np.shape(np.where(temp))[1]
        acc.append(float(true_amount) / float(global_amount))

    return acc


def train(train_path=None, val_path=None, record_path=None, batch_size=64, pre_learning_rate=0.0001, epoch_num=20,
          decay_rate=0.2, de_freq=20, regularization=0.0001, if_test=False, test_path=None, using_model=False,
          save_path='model_save', inc_learn=True, fold=0, avg_val_acc=None, avg_val_loss=None,
          avg_val_f1=None, last_fold=False, use_l2=True, clip=True, **kwargs):
    """Main part of the program."""

    # Value init #
    clip_value = 0.0625
    record_w = False
    change_op = False
    change_step = int(SAMPLE_NUM * 310 / batch_size)
    train_record_freq = 10
    rollback_is_act = False
    init_decay = 0.99
    print_bnorm = True
    print_grad = False
    # Dist init #
    init_train_scale = 0.016
    init_prior_scale = 0.1
    init_cut = 1 / 50

    tf.logging.set_verbosity(tf.logging.ERROR)

    print("  ------ START RUNNING ------")
    # Reading data for training and validation
    record = read_record(record_path)
    # pad_length = record['pad_length']
    train_length = record['train_length']
    # val_length = record['val_length']
    bucket_num = len(train_length)

    # Generate the index array
    train_length_ = train_length // batch_size + 1
    data_index = np.zeros(np.sum(train_length_))
    for get in range(len(train_length_) - 1):
        train_length_[get + 1] += train_length_[get]
    for num, tail in enumerate(train_length_):
        if num:
            data_index[train_length_[num - 1]:tail] += num
    data_index = data_index.astype(np.int64)
    np.random.shuffle(data_index)

    # Get inputs from different bucket
    train_inputs = [None for _ in range(bucket_num)]
    train_labels = [None for _ in range(bucket_num)]
    iterator = [None for _ in range(bucket_num)]
    for b_num in range(bucket_num):
        train_inputs[b_num], train_labels[b_num], iterator[b_num] = read_data(train_path[b_num], batch_size,
                                                                              is_train=True,
                                                                              length_arr=SEQUENCE_LENGTH,
                                                                              length_index=b_num)
    val_inputs, val_labels, val_iterator = read_data(val_path, batch_size, is_train=False)
    test_inputs, test_labels, test_iterator = read_data(test_path, batch_size, is_train=False)

    # session config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Establish session
    sess = tf.Session(config=config)

    # Batch normalization update mean and variance or not
    is_act = tf.Variable(1.0, trainable=False)  # 1.0=on, 0.0=off
    tf.add_to_collection('change', tf.assign(is_act, tf.Variable(0.0, trainable=False)))
    decay = tf.get_variable('decay', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(value=init_decay),
                            trainable=False)

    tf.add_to_collection('change', tf.assign(decay, 1.0))
    change_ops = tf.get_collection('change')
    # Show value or not
    if net_num == 1 or print_bnorm:
        print_bnorm_value = True
    else:
        print_bnorm_value = False

    # Learning rate decrease
    rate = 1.0
    learning_rate = tf.Variable(pre_learning_rate, dtype=tf.float32, trainable=False)

    # Respectively set loss and accuracy of training and validation.
    # get validation and test loss and so on, training part will been gotten later.
    train_loss = [None for _ in range(bucket_num)]
    train_accuracy = [None for _ in range(bucket_num)]
    train_ops = [None for _ in range(bucket_num)]

    # Choice of distillation or nor
    if kwargs['distillation']:
        print('  ------ DISTILLATION TRAINING ------')
        net_prior = net_simple
        net_train = net_binary

        train_outputs = [None for _ in range(bucket_num)]
        train_params = [None for _ in range(bucket_num)]
        test_bn_value = [None for _ in range(bucket_num)]

        # Outputs of binary net
        with tf.variable_scope('structure_b') as scope:
            _, train_outputs[0], train_params[0], layer_num, test_bn_value[0] = net_train.network_structure(
                train_inputs[0], is_training=True, is_act=is_act, decay=decay)
            scope.reuse_variables()
            for i in range(bucket_num - 1):
                _, train_outputs[i + 1], train_params[i + 1], _, test_bn_value[i + 1] = net_train.network_structure(
                    train_inputs[i + 1], is_training=True, is_act=is_act, decay=decay)
            _, val_outputs, _, _, _ = net_train.network_structure(val_inputs, is_training=False, is_act=is_act,
                                                                  decay=0.99)
            _, test_outputs, _, _, _ = net_train.network_structure(test_inputs, is_training=False, is_act=is_act,
                                                                   decay=0.99)

        # Get full precision network outputs
        model_outputs = [None for _ in range(bucket_num)]
        with tf.variable_scope('structure_0') as scope_:
            _, model_outputs[0], _ = net_prior.network_structure(train_inputs[0], is_training=False, is_act=is_act,
                                                                 decay=0.99)
            scope_.reuse_variables()
            for i in range(bucket_num - 1):
                _, model_outputs[i + 1], _ = net_prior.network_structure(train_inputs[i + 1], is_training=False,
                                                                         is_act=is_act, decay=0.99)

        # Trained net outputs scale
        train_scale = tf.get_variable("train_scale_value", shape=[1],
                                      initializer=tf.constant_initializer(value=init_train_scale),
                                      trainable=False)

        prior_scale = tf.get_variable("prior_scale_value", shape=[1],
                                      initializer=tf.constant_initializer(value=init_prior_scale),
                                      trainable=False)
        # Hard label loss constraint value
        hard_label_cut_value = tf.get_variable("hard_label_cut_value", shape=[1],
                                               initializer=tf.constant_initializer(value=init_cut),
                                               trainable=False)

        for op_num in range(bucket_num):
            train_loss[op_num], train_accuracy[op_num], train_ops[op_num] = get_op_dist(train_outputs[op_num],
                                                                                        train_labels[op_num],
                                                                                        model_outputs[op_num],
                                                                                        regularization,
                                                                                        learning_rate,
                                                                                        train_params[op_num],
                                                                                        use_l2=use_l2,
                                                                                        clip=clip,
                                                                                        prior_scale=prior_scale,
                                                                                        train_scale=train_scale,
                                                                                        hard_label=kwargs['hard_label'],
                                                                                        hard_label_cut_value=hard_label_cut_value)

        # Create saver
        train_param = get_variables_with_name(['structure_b'], train_only=False, delete='scale',
                                              name='get_train_variables')
        saver = tf.train.Saver(max_to_keep=1, var_list=train_param)
        prior_param = get_variables_with_name(['structure_0'], train_only=False, name='get_model_variables')
        saver_restore = tf.train.Saver(var_list=prior_param)
        tl.files.exists_or_mkdir(save_path)
        model_path = kwargs['model_path']

        scope_name = 'structure_b'
    else:
        train_outputs = [None for _ in range(bucket_num)]
        train_params = [None for _ in range(bucket_num)]
        test_bn_value = [None for _ in range(bucket_num)]

        with tf.variable_scope('structure_' + str(fold)) as scope:
            _, train_outputs[0], train_params[0], layer_num, test_bn_value[0] = net_struct.network_structure(
                train_inputs[0], is_training=True, is_act=is_act, decay=decay)
            scope.reuse_variables()
            for i in range(bucket_num - 1):
                _, train_outputs[i + 1], train_params[i + 1], _, test_bn_value[i + 1] = net_struct.network_structure(
                    train_inputs[i + 1], is_training=True, is_act=is_act, decay=decay)
            _, val_outputs, _, _, _ = net_struct.network_structure(val_inputs, is_training=False, is_act=is_act,
                                                                   decay=0.99)
            _, test_outputs, _, _, _ = net_struct.network_structure(test_inputs, is_training=False, is_act=is_act,
                                                                    decay=0.99)

        for op_num in range(bucket_num):
            train_loss[op_num], train_accuracy[op_num], train_ops[op_num] = get_op(train_outputs[op_num],
                                                                                   train_labels[op_num],
                                                                                   regularization,
                                                                                   learning_rate,
                                                                                   train_params[op_num],
                                                                                   use_l2=use_l2,
                                                                                   clip=clip)

        # Save model
        saver = tf.train.Saver(max_to_keep=1)
        tl.files.exists_or_mkdir(save_path)

        scope_name = 'structure_' + str(fold)

    val_loss, val_accuracy, val_logits = loss_calculate(val_outputs, val_labels, regularization, use_l2=use_l2,
                                                        name='val_loss')
    test_loss, test_accuracy, test_logits = loss_calculate(test_outputs, test_labels, regularization, use_l2=use_l2,
                                                           name='test_loss')

    grads_op = [None for _ in range(bucket_num)]
    all_grads_op = [None for _ in range(bucket_num)]
    for op_num in range(bucket_num):
        grads_op[op_num], all_grads_op[op_num] = get_gradients(train_loss[op_num], layer_num=layer_num,
                                                               variable_info=[scope_name, 'W'])

    # Judge whether using saved model.(undone)
    if using_model:
        ckpt = tf.train.get_checkpoint_state(save_path)
        dir_path = os.path.join(save_path, 'checkpoint')
        assert os.path.exists(dir_path), print('There is no model existed.')
        logging.info('Restore the model from checkpoint')
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = int(ckpt.model_checkpoint_path.split('-')[-1])
        start_step = global_step
        rebuild_learning = tf.assign(learning_rate, pre_learning_rate)
        sess.run(rebuild_learning)
        offset_fix = True
        if rollback_is_act:
            sess.run([tf.assign(is_act, 1.0), tf.assign(decay, init_decay)])
    else:
        global_step = 0
        sess.run(tf.global_variables_initializer())
        offset_fix = False
    # sess.run(tf.local_variables_initializer())

    if kwargs['distillation']:
        # Load trained model
        ckpt = tf.train.get_checkpoint_state(model_path)
        dir_path = os.path.join(model_path, 'checkpoint')
        assert os.path.exists(dir_path), print('There is no model existed.')
        print('Restore the model from checkpoint')
        # Restores from checkpoint
        saver_restore.restore(sess, ckpt.model_checkpoint_path)

    # Set up tensorboard summaries and saver
    # tl.files.exists_or_mkdir(log_dir)
    summary_writer = tf.summary.FileWriter('logs/part_' + str(fold), sess.graph)

    # Set up summary nodes
    for param in get_variables_with_name(['W', scope_name], name='histogram'):
        tf.summary.histogram(param.op.name, param)

    # Clip op
    for param_n, param in enumerate(get_variables_with_name(['W_conv1d', scope_name], delete='cnn_0',
                                                            name='clip')):
        tf.add_to_collection('clip', tf.assign(param, tf.clip_by_value(param, -clip_value, clip_value)))
    clip_op = tf.get_collection('clip')

    # The 0 index of moving mean is the data wanted, and the rest is Adam parameters
    if print_bnorm_value:
        moving_mean = get_variables_with_name([scope_name, '_0/moving_mean'], train_only=False, name='moving_mean')[0]
        # moving_mean = tf.norm(moving_mean, axis=1, ord=2)
        moving_var_ = \
            get_variables_with_name([scope_name, '_0/moving_variance'], train_only=False, name='moving_variance')[0]
        moving_var = tf.sqrt(moving_var_ + 0.00001)

    # Summary op
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summaries.add(tf.summary.scalar('sh_learning_rate_' + str(fold), learning_rate))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Print all variables
    _ = get_variables_with_name(['structure_'], train_only=False, name='param_table')

    # How many epoch to save to the tensorboard
    val_record_freq = train_record_freq
    if train_record_freq < 1:
        val_record_freq = 1
    step_freq = int((SAMPLE_NUM / batch_size) * train_record_freq)

    loss_ep = 0
    acc_ep = 0

    if if_test:
        print("  ------ Start testing the network ------")
        sess.run(test_iterator.initializer)
        test_loss_ep = 0
        test_acc_ep = 0
        test_step = 0

        test_epoch_logits = None
        test_epoch_labels = None
        f = 0
        start_time = time.time()
        while True:
            try:
                if f == 0:
                    time_in = time.time()
                test_loss_, test_accuracy_, test_outputs_, test_labels_, test_logits_, test_inputs_ = sess.run(
                    [test_loss, test_accuracy, test_outputs, test_labels, test_logits, test_inputs])

                if test_epoch_logits is None:
                    test_epoch_logits = np.argmax(test_outputs_, 1)
                else:
                    test_epoch_logits = np.hstack((test_epoch_logits, np.argmax(test_outputs_, 1)))
                    test_epoch_labels = np.hstack((test_epoch_labels, test_labels_))

                test_loss_ep += test_loss_
                test_acc_ep += test_accuracy_
                test_step += 1

                if f == 0:
                    f = 1
                    print("time", time.time() - time_in)

            except tf.errors.OutOfRangeError:
                break
        last_time = time.time() - start_time
        print("last time: %fs in %d data amounts" % (last_time, test_step))

        test_f1_ep_, f1_test = f1_measure(test_epoch_logits, test_epoch_labels)

        test_loss_ep_ = test_loss_ep / test_step
        test_acc_ep_ = test_acc_ep / test_step

        print("     Test result: loss %f - accuracy %f - f1 %f" % (test_loss_ep_, test_acc_ep_, test_f1_ep_))
    else:
        print("  ------ Start training the network ------   ")

        # decide which part of data to use
        for num in range(epoch_num):
            flag = 0
            is_first = True
            data_num = 0

            for ite in range(bucket_num):
                sess.run(iterator[ite].initializer)
            while data_num < len(data_index):
                # sess.run(iterator[data_index[data_num]].initializer)
                try:
                    # Get run operation
                    [loss, accuracy, train_op, outputs, labels, grads_norm, all_grads_norm, mean, var] = generate_op(
                        train_loss[data_index[data_num]],
                        train_accuracy[data_index[data_num]],
                        train_ops[data_index[data_num]],
                        train_outputs[data_index[data_num]],
                        train_labels[data_index[data_num]],
                        grads_op[data_index[data_num]],
                        all_grads_op[data_index[data_num]],
                        test_bn_value[data_index[data_num]][0],
                        test_bn_value[data_index[data_num]][1])

                    # Summary step, record the data per 5 epochs
                    if (global_step + 1) % step_freq == 0 and global_step != 0:
                        # record param
                        loss_, acc_, _, outputs_, labels_, result, grads_norm_, all_grads_norm_ = sess.run([
                            loss, accuracy, train_op, outputs, labels, summary_op, grads_norm, all_grads_norm])

                        if clip:
                            sess.run(clip_op)
                        summary_writer.add_summary(result, global_step)

                        loss_ep += loss_
                        acc_ep += acc_

                        global_step += 1

                        # record loss and acc
                        if offset_fix:
                            offset_step = global_step - start_step
                            loss_ep_ = loss_ep / offset_step
                            acc_ep_ = acc_ep / offset_step
                            offset_fix = False
                        else:
                            loss_ep_ = loss_ep / step_freq
                            acc_ep_ = acc_ep / step_freq

                        summary_list = [
                            tf.Summary.Value(tag="sh_training_accuracy_" + str(fold), simple_value=acc_ep_),
                            tf.Summary.Value(tag="sh_training_loss_" + str(fold), simple_value=loss_ep_),
                            tf.Summary.Value(tag="all_gradient_norm_" + str(fold), simple_value=all_grads_norm_)]

                        if print_grad:
                            for l in range(layer_num):
                                summary_list.append(
                                    tf.Summary.Value(tag="grads_norm_" + str(fold) + '_' + str(l),
                                                     simple_value=grads_norm_[l]))

                        if print_bnorm_value:
                            moving_mean_, moving_var_, mean_, var_ = sess.run([moving_mean, moving_var, mean, var])
                            moving_list = [
                                tf.Summary.Value(tag="bn_moving_mean_" + str(fold), simple_value=moving_mean_[0]),
                                tf.Summary.Value(tag="bn_moving_variance_" + str(fold), simple_value=moving_var_[0])]
                            current_list = [
                                tf.Summary.Value(tag="bn_mean_" + str(fold), simple_value=mean_[0]),
                                tf.Summary.Value(tag="bn_variance_" + str(fold), simple_value=var_[0])]
                            summary_list.extend(moving_list)
                            summary_list.extend(current_list)

                        summary = tf.Summary(value=summary_list)
                        summary_writer.add_summary(summary, global_step)

                        # renew loss
                        loss_ep = 0
                        acc_ep = 0

                    else:
                        # get training loss and acc
                        if record_w:
                            loss_, acc_, labels_, outputs_, _, out_, w_0_ = sess.run(
                                [loss, accuracy, labels, outputs, train_op, w_0])
                        else:
                            loss_, acc_, labels_, outputs_, _ = sess.run([loss, accuracy, labels, outputs, train_op])

                        if clip:
                            sess.run(clip_op)

                        loss_ep += loss_
                        acc_ep += acc_

                        global_step += 1

                    # Standard print
                    if global_step % 1000 == 0:
                        print("step {} - training loss {} - training accuracy {}".format(
                            global_step,
                            loss_,
                            acc_))

                    # Collect outputs for f1 calculating
                    if flag == 0:
                        epoch_logits = np.argmax(outputs_, 1)
                        epoch_labels = labels_
                        flag = 1
                    else:
                        epoch_logits = np.hstack((epoch_logits, np.argmax(outputs_, 1)))
                        epoch_labels = np.hstack((epoch_labels, labels_))

                    # Judge whether using slow-start
                    if inc_learn:
                        change_learning_rate = 0.001
                        inc_num = 3
                        if (num + 1) == inc_num:
                            de_learning_op = tf.assign(learning_rate, change_learning_rate)
                            sess.run(de_learning_op)

                        if (num + 1) % de_freq == 0 and is_first:
                            rate = rate * decay_rate
                            de_learning_op = tf.assign(learning_rate, change_learning_rate * rate)
                            sess.run(de_learning_op)
                            is_first = False
                    else:
                        if (num + 1) % de_freq == 0 and is_first:
                            rate = rate * decay_rate
                            de_learning_op = tf.assign(learning_rate, pre_learning_rate * rate)
                            sess.run(de_learning_op)
                            is_first = False

                    # Judge whether activate mean and variance calculation
                    if change_op:
                        if global_step == change_step:
                            sess.run(change_ops)
                            print('Stop batch norm')

                    data_num += 1

                except tf.errors.OutOfRangeError:
                    break

            # calculate the epoch f1
            epoch_f1, f1 = f1_measure(epoch_logits, epoch_labels)

            # save model
            saver.save(sess, save_path + '/model_' + str(fold) + '.ckpt', global_step=global_step)

            # execute validation operation per 10 epochs
            if (num + 1) % val_record_freq == 0:
                # get validation loss and acc
                val_loss_ep = 0
                val_acc_ep = 0
                val_step = 0

                val_epoch_logits = None
                val_epoch_labels = None

                sess.run(val_iterator.initializer)
                while True:
                    try:
                        val_loss_, val_accuracy_, val_outputs_, val_labels_, val_logits_, val_inputs_ = sess.run(
                            [val_loss, val_accuracy, val_outputs, val_labels, val_logits, val_inputs])

                        # val_f1_ = _f1_measure(val_outputs_, val_labels_)
                        if val_epoch_logits is None:
                            val_epoch_logits = np.argmax(val_outputs_, 1)
                            val_epoch_labels = val_labels_
                        else:
                            val_epoch_logits = np.hstack((val_epoch_logits, np.argmax(val_outputs_, 1)))
                            val_epoch_labels = np.hstack((val_epoch_labels, val_labels_))
                        val_loss_ep += val_loss_
                        val_acc_ep += val_accuracy_
                        val_step += 1

                    except tf.errors.OutOfRangeError:
                        break

                val_f1_ep, f1_ = f1_measure(val_epoch_logits, val_epoch_labels)

                val_loss_ep_ = val_loss_ep / val_step
                val_acc_ep_ = val_acc_ep / val_step

                # record acc loss and acc
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="sh_val_accuracy_" + str(fold), simple_value=val_acc_ep_),
                    tf.Summary.Value(tag="sh_val_loss_" + str(fold), simple_value=val_loss_ep_),
                    tf.Summary.Value(tag="sh_val_f1_" + str(fold), simple_value=val_f1_ep),
                    tf.Summary.Value(tag="sh_training_f1_" + str(fold), simple_value=epoch_f1),
                    tf.Summary.Value(tag="detail_train_f1_N_" + str(fold), simple_value=f1[0]),
                    tf.Summary.Value(tag="detail_train_f1_A_" + str(fold), simple_value=f1[1]),
                    tf.Summary.Value(tag="detail_train_f1_O_" + str(fold), simple_value=f1[2]),
                    tf.Summary.Value(tag="detail_train_f1_~_" + str(fold), simple_value=f1[3]),
                    tf.Summary.Value(tag="detail_val_f1_N_" + str(fold), simple_value=f1_[0]),
                    tf.Summary.Value(tag="detail_val_f1_A_" + str(fold), simple_value=f1_[1]),
                    tf.Summary.Value(tag="detail_val_f1_O_" + str(fold), simple_value=f1_[2]),
                    tf.Summary.Value(tag="detail_val_f1_~_" + str(fold), simple_value=f1_[3])
                ])
                summary_writer.add_summary(summary, global_step)

                avg_val_acc.append(val_acc_ep_)
                avg_val_loss.append(val_loss_ep_)
                avg_val_f1.append(val_f1_ep)

        # Get the average value of same step in validation and record it
        if last_fold is True:
            avg_val_acc_ = np.array(avg_val_acc) / float(fold)
            avg_val_loss_ = np.array(avg_val_loss) / float(fold)
            avg_val_f1_ = np.array(avg_val_f1) / float(fold)

            for re in range(len(avg_val_acc_)):
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="avg_val_acc", simple_value=avg_val_acc_[re]),
                    tf.Summary.Value(tag="avg_val_loss", simple_value=avg_val_loss_[re]),
                    tf.Summary.Value(tag="avg_val_f1", simple_value=avg_val_f1_[re])
                ])
                summary_writer.add_summary(summary, re)

    summary_writer.close()
    sess.close()

    return np.array(avg_val_acc), np.array(avg_val_loss), np.array(avg_val_f1)
