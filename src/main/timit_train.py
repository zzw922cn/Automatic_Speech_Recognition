#-*- coding:utf-8 -*-
#!/usr/bin/python
''' Automatic Speech Recognition for TIMIT corpus

author(s):
zzw922cn
     
date:2017-4-15
'''

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import time
import datetime
import os
from six.moves import cPickle
from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.contrib.rnn.python.ops import rnncell
from tensorflow.contrib.rnn.python.ops.core_rnn import static_bidirectional_rnn

from utils.utils import load_batched_data
from utils.utils import describe
from utils.utils import getAttrs
from utils.utils import output_to_sequence
from utils.utils import list_dirs
from utils.utils import logging
from utils.utils import count_params
from utils.utils import target2phoneme
from utils.utils import get_edit_distance
from utils.taskUtils import get_num_classes
from utils.taskUtils import check_path_exists
from utils.taskUtils import dotdict

from models.resnet import ResNet
from models.brnn import BiRNN
from models.dynamic_brnn import DBiRNN

from tensorflow.python.platform import flags
from tensorflow.python.platform import app

model_functions_dict = {'ResNet': ResNet, 'BiRNN': BiRNN, 'DBiRNN': DBiRNN}

activation_functions_dict = {
    'sigmoid': tf.sigmoid, 'tanh': tf.tanh, 'relu': tf.nn.relu, 'relu6': tf.nn.relu6, 
    'elu': tf.nn.elu, 'softplus': tf.nn.softplus, 'softsign': tf.nn.softsign
    # for detailed intro, go to https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_
    }
optimizer_functions_dict = {
    'gd': tf.train.GradientDescentOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adam': tf.train.AdamOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer
    } 
    

flags.DEFINE_integer('tf_random_seed', 123456, 'set the random seed for tf graph')
flags.DEFINE_integer('np_random_seed', 123456, 'set the random seed for tf graph')

flags.DEFINE_string('level', 'phn', 'set the task level, phn, cha, or seq2seq, seq2seq will be supported soon')
flags.DEFINE_string('model', 'DBiRNN', 'set the model to use, DBiRNN, BiRNN, ResNet..')
flags.DEFINE_string('rnncell', 'gru', 'set the rnncell to use, rnn, gru, lstm...')
flags.DEFINE_integer('num_layer', 2, 'set the layers for rnn')
flags.DEFINE_string('activation', 'relu', 'set the activation to use, sigmoid, tanh, relu, elu...')
flags.DEFINE_string('optimizer', 'adam', 'set the optimizer to use, sgd, adam...')

flags.DEFINE_integer('batch_size', 64, 'set the batch size')
flags.DEFINE_integer('num_hidden', 128, 'set the hidden size of rnn cell')
flags.DEFINE_integer('num_feature', 39, 'set the size of input feature')
flags.DEFINE_integer('num_classes', 30, 'set the number of output classes')
flags.DEFINE_integer('num_epochs', 1, 'set the number of epochs')
flags.DEFINE_float('lr', 0.0001, 'set the learning rate')
flags.DEFINE_float('dropout_prob', 0.1, 'set probability of dropout')
flags.DEFINE_float('grad_clip', 1.0, 'set the threshold of gradient clipping')
flags.DEFINE_string('datadir', '/home/pony/github/data/timit', 'set the data root directory')
flags.DEFINE_string('logdir', '/home/pony/github/log/timit', 'set the log directory')
flags.DEFINE_string('mode', 'train', 'set whether to train or test')
flags.DEFINE_boolean('keep', False, 'set whether to restore a model')

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.tf_random_seed)
np.random.seed(FLAGS.np_random_seed)

level = FLAGS.level
model_fn = model_functions_dict[FLAGS.model]
rnncell = FLAGS.rnncell
num_layer = FLAGS.num_layer

activation_fn = activation_functions_dict[FLAGS.activation]
optimizer_fn = optimizer_functions_dict[FLAGS.optimizer]

batch_size = FLAGS.batch_size
time_length = FLAGS.time_length
num_hidden = FLAGS.num_hidden
num_feature = FLAGS.num_feature
num_classes = get_num_classes(level)
num_epochs = FLAGS.num_epochs
lr = FLAGS.lr
grad_clip = FLAGS.grad_clip
datadir = FLAGS.datadir

logdir = FLAGS.logdir
savedir = os.path.join(logdir, level, 'save')
resultdir = os.path.join(logdir, level, 'result')
loggingdir = os.path.join(logdir, level, 'logging')
check_path_exists([logdir, savedir, resultdir, loggingdir])

mode = FLAGS.mode
keep = FLAGS.keep
keep_prob = 1-FLAGS.dropout_prob

if inference is True:
  print 'Inference Mode...'
  batch_size = 100
  num_epochs = 1
train_mfcc_dir = os.path.join(datadir, level, 'train', 'mfcc')
train_label_dir = os.path.join(datadir, level, 'train', 'label')
test_mfcc_dir = os.path.join(datadir, level, 'test', 'mfcc')
test_label_dir = os.path.join(datadir, level, 'test', 'label')

logfile = os.path.join(loggingdir, str(
    datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S') + '.txt').replace(' ', '').replace(
    '/', ''))


class Trainer(object):

    @describe
    def load_data(self, args, mode, type):
        if mode == 'train':
            return load_batched_data(train_mfcc_dir, train_label_dir, batch_size, mode, type)
        elif mode == 'test':
            batch_size = test_batch_size
            return load_batched_data(test_mfcc_dir, test_label_dir, test_batch_size, mode, type)
        else:
            raise TypeError('mode should be train or test.')

    def train(self):
        # load data
        args_dict = {'maxTimeSteps': maxTimeSteps,
                     'rnncell': rnncell,
                     'batch_size', batch_size,   
        args = self.args
        batchedData, maxTimeSteps, totalN = self.load_data(args, mode='train', type=level)
        model = model_fn(args, maxTimeSteps)

        # count the num of params
        num_params = count_params(model, mode='trainable')
        all_num_params = count_params(model, mode='all')
        model.config['trainable params'] = num_params
        model.config['all params'] = all_num_params
        print(model.config)

        with tf.Session(graph=model.graph) as sess:
            # restore from stored model
            if keep == True:
                ckpt = tf.train.get_checkpoint_state(savedir)
                if ckpt and ckpt.model_checkpoint_path:
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Model restored from:' + savedir)
            else:
                print('Initializing')
                sess.run(model.initial_op)

            for epoch in range(num_epochs):
                ## training
                start = time.time()
                print('Epoch', epoch + 1, '...')
                batchErrors = np.zeros(len(batchedData))
                batchRandIxs = np.random.permutation(len(batchedData))

                for batch, batchOrigI in enumerate(batchRandIxs):
                    batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                    batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                    feedDict = {model.inputX: batchInputs, model.targetIxs: batchTargetIxs,
                                model.targetVals: batchTargetVals, model.targetShape: batchTargetShape,
                                model.seqLengths: batchSeqLengths}

                    if level == 'cha':
                        _, l, pre, y, er = sess.run([model.optimizer, model.loss,
                                                     model.predictions,
                                                     model.targetY,
                                                     model.errorRate],
                                                    feed_dict=feedDict)
                        batchErrors[batch] = er
                        print('\ntotal:{},batch:{}/{},epoch:{}/{},loss={:.3f},mean CER={:.3f}\n'.format(
                            totalN,
                            batch + 1,
                            len(batchRandIxs),
                            epoch + 1,
                            num_epochs,
                            l,
                            er / batch_size))

                    elif level == 'phn':
                        _, l, pre, y = sess.run([model.optimizer, model.loss,
                                                 model.predictions,
                                                 model.targetY],
                                                feed_dict=feedDict)
                        er = get_edit_distance([pre.values], [y.values], True, 'train', level)
                        print('\ntotal:{},batch:{}/{},epoch:{}/{},loss={:.3f},mean PER={:.3f}\n'.format(
                            totalN,
                            batch + 1,
                            len(batchRandIxs),
                            epoch + 1,
                            num_epochs,
                            l,
                            er / batch_size))
                        batchErrors[batch] = er * len(batchSeqLengths)

                    # NOTE:
                    if er / batch_size == 1.0:
                        break

                    if batch % 30 == 0:
                        print('Truth:\n' + output_to_sequence(y, type=level))
                        print('Output:\n' + output_to_sequence(pre, type=level))

                    if (save == True) and ((epoch * len(batchRandIxs) + batch + 1) % 20 == 0 or (
                                    epoch == num_epochs - 1 and batch == len(batchRandIxs) - 1)):
                        checkpoint_path = os.path.join(savedir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                        print('Model has been saved in file')
                end = time.time()
                delta_time = end - start
                print('Epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) + ' s')

                if save == True and (epoch + 1) % 1 == 0:
                    checkpoint_path = os.path.join(savedir, 'model.ckpt')
                    model.saver.save(sess, checkpoint_path, global_step=epoch)
                    print('Model has been saved in file')
                epochER = batchErrors.sum() / totalN
                print('Epoch', epoch + 1, 'mean train error rate:', epochER)
                logging(model, self.logfile, epochER, epoch, delta_time, mode='config')
                logging(model, self.logfile, epochER, epoch, delta_time, mode='train')

    def test(self):
        # load data
        args = self.args
        batchedData, maxTimeSteps, totalN = self.load_data(args, mode='test', type=level)
        model = model_functions[model](args, maxTimeSteps)

        num_params = count_params(model, mode='trainable')
        all_num_params = count_params(model, mode='all')
        model.config['trainable params'] = num_params
        model.config['all params'] = all_num_params
        with tf.Session(graph=model.graph) as sess:
            ckpt = tf.train.get_checkpoint_state(savedir)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model restored from:' + savedir)

            batchErrors = np.zeros(len(batchedData))
            batchRandIxs = np.random.permutation(len(batchedData))
            for batch, batchOrigI in enumerate(batchRandIxs):
                batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                feedDict = {model.inputX: batchInputs,
                            model.targetIxs: batchTargetIxs,
                            model.targetVals: batchTargetVals,
                            model.targetShape: batchTargetShape,
                            model.seqLengths: batchSeqLengths}

                if level == 'cha':
                    l, pre, y, er = sess.run([model.loss,
                                              model.predictions,
                                              model.targetY,
                                              model.errorRate],
                                             feed_dict=feedDict)
                    batchErrors[batch] = er
                    print('\ntotal:{},batch:{}/{},loss={:.3f},mean CER={:.3f}\n'.format(
                        totalN,
                        batch + 1,
                        len(batchRandIxs),
                        l,
                        er / batch_size))

                elif level == 'phn':
                    l, pre, y = sess.run([model.loss,
                                          model.predictions,
                                          model.targetY],
                                         feed_dict=feedDict)
                    er = get_edit_distance([pre.values], [y.values], True, 'test', level)
                    print('\ntotal:{},batch:{}/{},loss={:.3f},mean PER={:.3f}\n'.format(
                        totalN,
                        batch + 1,
                        len(batchRandIxs),
                        l,
                        er / batch_size))
                    batchErrors[batch] = er * len(batchSeqLengths)

                print('Truth:\n' + output_to_sequence(y, type=level))
                print('Output:\n' + output_to_sequence(pre, type=level))

                with open(os.path.join(resultdir, task + '_result.txt'), 'a') as result:
                    result.write(output_to_sequence(y, type=level) + '\n')
                    result.write(output_to_sequence(pre, type=level) + '\n')
                    result.write('\n')
            epochER = batchErrors.sum() / totalN
            print(task + ' test error rate:', epochER)
            logging(model, self.logfile, epochER, mode='test')


if __name__ == '__main__':
  tr = Trainer()
  print(tr.mode + ' mode')
  if tr.mode == 'train':
      tr.train()
  elif tr.mode == 'test':
      tr.test()
