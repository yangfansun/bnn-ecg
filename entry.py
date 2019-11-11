"""Using this file to train network"""
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
import train

train_file = train

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('load_dir', 'E:\Study\Dataset\MIT-BIH\\2017\\5_bucket', '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'define the learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 2, 'define the learning rate.')
tf.app.flags.DEFINE_bool('if_test', False, 'define whether testing the model.')
tf.app.flags.DEFINE_integer('epoch_num', 1, 'define the number of single datas epoch')
tf.app.flags.DEFINE_float('decay_rate', 0.2, '')
tf.app.flags.DEFINE_integer('decay_frequency', 200, '')
tf.app.flags.DEFINE_float('regularization', 0.0002, '')
tf.app.flags.DEFINE_bool('using_model', False, 'whether using saved model.')
tf.app.flags.DEFINE_bool('slow_start', False, 'whether using slow pattern.')
tf.app.flags.DEFINE_integer('fold', 0, 'how many folds of cross-validation')
tf.app.flags.DEFINE_bool('use_l2', True, '')
tf.app.flags.DEFINE_bool('clip', False, '')
tf.app.flags.DEFINE_bool('hard_label', False, '')
tf.app.flags.DEFINE_bool('distillation', True, '')
tf.app.flags.DEFINE_string('model_path', 'model_save', '')


def main():
    # training entry

    # variables
    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    decay_rate = FLAGS.decay_rate
    de_freq = FLAGS.decay_frequency
    epoch_num = FLAGS.epoch_num
    if_test = FLAGS.if_test

    avg_val_acc = []
    avg_val_loss = []
    avg_val_f1 = []
    last_fold = False

    # for f in range(FLAGS.fold):
    train_path = []
    f = FLAGS.fold
    for i in range(5):
        train_path.append(os.path.join(FLAGS.load_dir, 'train_' + str(i) + '.tfrecord'))
    val_path = os.path.join(FLAGS.load_dir, 'val.tfrecord')
    test_path = os.path.join(FLAGS.load_dir, 'val.tfrecord')
    record_path = os.path.join(FLAGS.load_dir, 'record.tfrecord')
    avg_val_acc, avg_val_loss, avg_val_f1 = train_file.train(train_path,
                                                             val_path,
                                                             record_path,
                                                             batch_size,
                                                             learning_rate,
                                                             epoch_num,
                                                             decay_rate=decay_rate,
                                                             de_freq=de_freq,
                                                             regularization=FLAGS.regularization,
                                                             if_test=if_test,
                                                             test_path=test_path,
                                                             using_model=FLAGS.using_model,
                                                             save_path='model_save',
                                                             inc_learn=FLAGS.slow_start,
                                                             fold=f,
                                                             avg_val_acc=avg_val_acc,
                                                             avg_val_loss=avg_val_loss,
                                                             avg_val_f1=avg_val_f1,
                                                             last_fold=last_fold,
                                                             use_l2=FLAGS.use_l2,
                                                             clip=FLAGS.clip,
                                                             hard_label=FLAGS.hard_label,
                                                             distillation=FLAGS.distillation,
                                                             model_path=FLAGS.model_path)

    # record average accuracy of k-fold
    summary = np.concatenate((np.reshape(avg_val_acc, [1, -1]), np.reshape(avg_val_loss, [1, -1])), axis=0)
    summary = np.concatenate((summary, np.reshape(avg_val_f1, [1, -1])), axis=0)
    if not os.path.exists("avg"):
        os.mkdir("avg")
    np.save(os.path.join("avg", "avg_result_" + str(f)), summary)


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
