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

import argparse
import time
import datetime
import os
from six.moves import cPickle
from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.contrib.rnn.python.ops import rnn_cell
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

from models.resnet import ResNet
from models.brnn import BiRNN
from models.dynamic_brnn import DBiRNN

from tensorflow.python.platform import flags
from tensorflow.python.platform import app

model_functions = {'ResNet': ResNet, 'BiRNN': BiRNN, 'DBiRNN': DBiRNN}

flags.DEFINE_integer('tf_random_seed', 123456, 'set the random seed for tf graph')
flags.DEFINE_integer('np_random_seed', 123456, 'set the random seed for tf graph')

flags.DEFINE_integer('level', 'phn', 'set the task level, phn, cha, or seq2seq, seq2seq will be supported soon')

flags.DEFINE_integer('batch_size', 64, 'set the batch size')
flags.DEFINE_integer('num_hidden', 128, 'set the hidden size of rnn cell')
flags.DEFINE_integer('num_classes', 30, 'set the number of output classes')
flags.DEFINE_integer('num_epochs', 1, 'set the number of epochs')
flags.DEFINE_float('lr', 0.0001, 'set the learning rate')
flags.DEFINE_string('datadir', '/home/pony/github/data/timit', 'set the data root directory')
flags.DEFINE_string('logdir', '/home/pony/github/log/timit', 'set the log directory')
flags.DEFINE_boolean('mode', 'train', 'set whether to train or test')
flags.DEFINE_boolean('keep', False, 'set whether to restore a model')

FLAGS = flags.FLAGS

tf_random_seed = FLAGS.tf_random_seed
np_random_seed = FLAGS.np_random_seed

level = FLAGS.level
batch_size = FLAGS.batch_size
time_length = FLAGS.time_length
num_hidden = FLAGS.num_hidden
num_classes = get_num_classes(level)
num_epochs = FLAGS.num_epochs
lr = FLAGS.lr
datadir = FLAGS.datadir
logdir = FLAGS.logdir
savedir = os.path.join(logdir, level, 'save')
attention_mode = FLAGS.attention_mode
inference = FLAGS.inference
decoder_hidden_size = FLAGS.decoder_hidden_size
mode = FLAGS.mode
keep = FLAGS.keep


check_path_exists([logdir, save_dir])

if inference is True:
  print 'Inference Mode...'
  batch_size = 100
  num_epochs = 1

tf.set_random_seed(tf_random_seed)
np.random.seed(np_random_seed)

class Trainer(object):
    def __init__(self):

        self.train_mfcc_dir = os.path.join(datadir, level, 'train', 'mfcc')
        self.train_label_dir = os.path.join(datadir, level, 'train', 'label')
        self.test_mfcc_dir = os.path.join(datadir, level, 'test', 'mfcc')
        self.test_label_dir = os.path.join(datadir, level, 'test', 'label')


        parser.add_argument('--log_dir', type=str, default=log_dir[cat],
                            help='directory to log events while training')

        parser.add_argument('--model', default='DBiRNN',
                            help='model for ASR:DBiRNN,BiRNN,ResNet,...')

        parser.add_argument('--keep_prob', type=float, default=1,
                            help='set the keep probability of layer for dropout')

        parser.add_argument('--rnncell', type=str, default='gru',
                            help='rnn cell, 3 choices:rnn,lstm,gru')

        parser.add_argument('--num_layer', type=int, default=2,
                            help='set the number of hidden layer or bidirectional layer')

        parser.add_argument('--activation', default=tf.nn.elu,
                            help='set the activation function of each layer')

        parser.add_argument('--optimizer', type=type, default=tf.train.AdamOptimizer,
                            help='set the optimizer to train the model,eg:AdamOptimizer,GradientDescentOptimizer')

        parser.add_argument('--grad_clip', default=0.8,
                            help='set gradient clipping when backpropagating errors')

        parser.add_argument('--save', type=bool, default=True,
                            help='to save the model in the disk')

        parser.add_argument('--learning_rate', type=float, default=0.0001,
                            help='set the step size of each iteration')

        parser.add_argument('--num_epoch', type=int, default=1,
                            help='set the total number of training epochs')

        parser.add_argument('--batch_size', type=int, default=32,
                            help='set the number of training samples in a mini-batch')

        parser.add_argument('--test_batch_size', type=int, default=256,
                            help='set the number of testing samples in a mini-batch')

        parser.add_argument('--num_feature', type=int, default=39,
                            help='set the dimension of feature, ie: 39 mfccs, you can set 39 ')

        parser.add_argument('--num_hidden', type=int, default=num_hidden[cat],
                            help='set the number of neurons in hidden layer')

        parser.add_argument('--num_class', type=int, default=num_class[cat],
                            help='set the number of labels in the output layer, if timit phonemes, it is 62; if timit characters, it is 29; if libri characters, it is 29')

        parser.add_argument('--save_dir', type=str, default=save_dir[cat],
                            help='set the directory to save the model, containing checkpoint file and parameter file')

        parser.add_argument('--model_checkpoint_path', type=str, default=save_dir[cat],
                            help='set the directory to restore the model, containing checkpoint file and parameter file')

        self.args = parser.parse_args()

        self.logfile = self.args.log_dir + str(
            datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S') + '.txt').replace(' ', '').replace(
            '/', '')

    @describe
    def load_data(self, args, mode, type):
        if mode == 'train':
            return load_batched_data(args.train_mfcc_dir, args.train_label_dir, args.batch_size, mode, type)
        elif mode == 'test':
            args.batch_size = args.test_batch_size
            return load_batched_data(args.test_mfcc_dir, args.test_label_dir, args.test_batch_size, mode, type)
        else:
            raise TypeError('mode should be train or test.')

    def train(self):
        # load data
        args = self.args
        batchedData, maxTimeSteps, totalN = self.load_data(args, mode='train', type=args.level)
        model = model_functions[args.model](args, maxTimeSteps)

        # count the num of params
        num_params = count_params(model, mode='trainable')
        all_num_params = count_params(model, mode='all')
        model.config['trainable params'] = num_params
        model.config['all params'] = all_num_params
        print(model.config)

        with tf.Session(graph=model.graph) as sess:
            # restore from stored model
            if args.keep == True:
                ckpt = tf.train.get_checkpoint_state(args.save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Model restored from:' + args.save_dir)
            else:
                print('Initializing')
                sess.run(model.initial_op)

            for epoch in range(args.num_epoch):
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

                    if args.level == 'cha':
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
                            args.num_epoch,
                            l,
                            er / args.batch_size))

                    elif args.level == 'phn':
                        _, l, pre, y = sess.run([model.optimizer, model.loss,
                                                 model.predictions,
                                                 model.targetY],
                                                feed_dict=feedDict)
                        er = get_edit_distance([pre.values], [y.values], True, 'train', args.level)
                        print('\ntotal:{},batch:{}/{},epoch:{}/{},loss={:.3f},mean PER={:.3f}\n'.format(
                            totalN,
                            batch + 1,
                            len(batchRandIxs),
                            epoch + 1,
                            args.num_epoch,
                            l,
                            er / args.batch_size))
                        batchErrors[batch] = er * len(batchSeqLengths)

                    # NOTE:
                    if er / args.batch_size == 1.0:
                        break

                    if batch % 30 == 0:
                        print('Truth:\n' + output_to_sequence(y, type=args.level))
                        print('Output:\n' + output_to_sequence(pre, type=args.level))

                    if (args.save == True) and ((epoch * len(batchRandIxs) + batch + 1) % 20 == 0 or (
                                    epoch == args.num_epoch - 1 and batch == len(batchRandIxs) - 1)):
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                        print('Model has been saved in file')
                end = time.time()
                delta_time = end - start
                print('Epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) + ' s')

                if args.save == True and (epoch + 1) % 1 == 0:
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    model.saver.save(sess, checkpoint_path, global_step=epoch)
                    print('Model has been saved in file')
                epochER = batchErrors.sum() / totalN
                print('Epoch', epoch + 1, 'mean train error rate:', epochER)
                logging(model, self.logfile, epochER, epoch, delta_time, mode='config')
                logging(model, self.logfile, epochER, epoch, delta_time, mode='train')

    def test(self):
        # load data
        args = self.args
        batchedData, maxTimeSteps, totalN = self.load_data(args, mode='test', type=args.level)
        model = model_functions[args.model](args, maxTimeSteps)

        num_params = count_params(model, mode='trainable')
        all_num_params = count_params(model, mode='all')
        model.config['trainable params'] = num_params
        model.config['all params'] = all_num_params
        with tf.Session(graph=model.graph) as sess:
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model restored from:' + args.save_dir)

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

                if args.level == 'cha':
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
                        er / args.batch_size))

                elif args.level == 'phn':
                    l, pre, y = sess.run([model.loss,
                                          model.predictions,
                                          model.targetY],
                                         feed_dict=feedDict)
                    er = get_edit_distance([pre.values], [y.values], True, 'test', args.level)
                    print('\ntotal:{},batch:{}/{},loss={:.3f},mean PER={:.3f}\n'.format(
                        totalN,
                        batch + 1,
                        len(batchRandIxs),
                        l,
                        er / args.batch_size))
                    batchErrors[batch] = er * len(batchSeqLengths)

                print('Truth:\n' + output_to_sequence(y, type=args.level))
                print('Output:\n' + output_to_sequence(pre, type=args.level))

                '''
                l, pre, y = sess.run([ model.loss,
                                        model.predictions,
                                        model.targetY],
                                        feed_dict=feedDict)


                er = get_edit_distance([pre.values], [y.values], True, 'test', args.level)
                print(output_to_sequence(y,type=args.level))
                print(output_to_sequence(pre,type=args.level))
                '''
                with open(args.task + '_result.txt', 'a') as result:
                    result.write(output_to_sequence(y, type=args.level) + '\n')
                    result.write(output_to_sequence(pre, type=args.level) + '\n')
                    result.write('\n')
            epochER = batchErrors.sum() / totalN
            print(args.task + ' test error rate:', epochER)
            logging(model, self.logfile, epochER, mode='test')


if __name__ == '__main__':
  tr = Trainer()
  print(tr.args.mode + ' mode')
  if tr.args.mode == 'train':
      tr.train()
  elif tr.args.mode == 'test':
      tr.test()
