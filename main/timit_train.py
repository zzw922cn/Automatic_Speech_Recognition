#-*- coding:utf-8 -*-
#!/usr/bin/python
''' Automatic Speech Recognition for TIMIT corpus

Support for LibriSpeech will come soon. 
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
from utils.functionDictUtils import model_functions_dict
from utils.functionDictUtils import activation_functions_dict
from utils.functionDictUtils import optimizer_functions_dict


from tensorflow.python.platform import flags
from tensorflow.python.platform import app
    
flags.DEFINE_string('task', 'timit', 'set task name of this program')
flags.DEFINE_string('mode', 'train', 'set whether to train or test')
flags.DEFINE_boolean('keep', False, 'set whether to restore a model, when test mode, keep should be set to True')
flags.DEFINE_string('level', 'phn', 'set the task level, phn, cha, or seq2seq, seq2seq will be supported soon')
flags.DEFINE_string('model', 'DBiRNN', 'set the model to use, DBiRNN, BiRNN, ResNet..')
flags.DEFINE_string('rnncell', 'lstm', 'set the rnncell to use, rnn, gru, lstm...')
flags.DEFINE_integer('num_layer', 2, 'set the layers for rnn')
flags.DEFINE_string('activation', 'tanh', 'set the activation to use, sigmoid, tanh, relu, elu...')
flags.DEFINE_string('optimizer', 'adam', 'set the optimizer to use, sgd, adam...')
flags.DEFINE_boolean('layerNormalization', False, 'set whether to apply layer normalization to rnn cell')

flags.DEFINE_integer('batch_size', 32, 'set the batch size')
flags.DEFINE_integer('num_hidden', 256, 'set the hidden size of rnn cell')
flags.DEFINE_integer('num_feature', 39, 'set the size of input feature')
flags.DEFINE_integer('num_classes', 30, 'set the number of output classes')
flags.DEFINE_integer('num_epochs', 500, 'set the number of epochs')
flags.DEFINE_float('lr', 0.0001, 'set the learning rate')
flags.DEFINE_float('dropout_prob', 0.1, 'set probability of dropout')
flags.DEFINE_float('grad_clip', 1, 'set the threshold of gradient clipping, -1 denotes no clipping')
flags.DEFINE_string('datadir', '/home/pony/github/data/timit', 'set the data root directory')
flags.DEFINE_string('logdir', '/home/pony/github/log/timit', 'set the log directory')


FLAGS = flags.FLAGS

task = FLAGS.task
level = FLAGS.level
model_fn = model_functions_dict[FLAGS.model]
rnncell = FLAGS.rnncell
num_layer = FLAGS.num_layer

activation_fn = activation_functions_dict[FLAGS.activation]
optimizer_fn = optimizer_functions_dict[FLAGS.optimizer]

batch_size = FLAGS.batch_size
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

print('%s mode...'%str(mode))
if mode == 'test':
  batch_size = 100
  num_epochs = 1

train_mfcc_dir = os.path.join(datadir, level, 'train', 'mfcc')
train_label_dir = os.path.join(datadir, level, 'train', 'label')
test_mfcc_dir = os.path.join(datadir, level, 'test', 'mfcc')
test_label_dir = os.path.join(datadir, level, 'test', 'label')
logfile = os.path.join(loggingdir, str(datetime.datetime.strftime(datetime.datetime.now(), 
    '%Y-%m-%d %H:%M:%S') + '.txt').replace(' ', '').replace('/', ''))


class Runner(object):

    def _default_configs(self):
      return {'level': level,
              'rnncell': rnncell,
              'batch_size': batch_size,
              'num_hidden': num_hidden,
              'num_feature': num_feature,
              'num_class': num_classes,
              'num_layer': num_layer,
              'activation': activation_fn,
              'optimizer': optimizer_fn,
              'learning_rate': lr,
              'keep_prob': keep_prob,
              'grad_clip': grad_clip,
            }

    @describe
    def load_data(self, args, mode, type):
        if mode == 'train':
            return load_batched_data(train_mfcc_dir, train_label_dir, batch_size, mode, type)
        elif mode == 'test':
            return load_batched_data(test_mfcc_dir, test_label_dir, batch_size, mode, type)
        else:
            raise TypeError('mode should be train or test.')

    def run(self):
        # load data
        args_dict = self._default_configs()
        args = dotdict(args_dict)
        batchedData, maxTimeSteps, totalN = self.load_data(args, mode=mode, type=level)
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
                if mode == 'train':
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
                        if mode == 'train':
                            _, l, pre, y, er = sess.run([model.optimizer, model.loss,
                                model.predictions, model.targetY, model.errorRate],
                                feed_dict=feedDict)

                            batchErrors[batch] = er
                            print('\n{} mode, total:{},batch:{}/{},epoch:{}/{},train loss={:.3f},mean train CER={:.3f}\n'.format(
                                level, totalN, batch+1, len(batchRandIxs), epoch+1, num_epochs, l, er/batch_size))

                        elif mode == 'test':
                            l, pre, y, er = sess.run([model.loss, model.predictions, 
                                model.targetY, model.errorRate], feed_dict=feedDict)
                            batchErrors[batch] = er
                            print('\n{} mode, total:{},batch:{}/{},test loss={:.3f},mean test CER={:.3f}\n'.format(
                                level, totalN, batch+1, len(batchRandIxs), l, er/batch_size))

                    elif level == 'phn':
                        if mode == 'train':
                            _, l, pre, y = sess.run([model.optimizer, model.loss,
                                model.predictions, model.targetY],
                                feed_dict=feedDict)
                  
                            er = get_edit_distance([pre.values], [y.values], True, level)
                            print('\n{} mode, total:{},batch:{}/{},epoch:{}/{},train loss={:.3f},mean train PER={:.3f}\n'.format(
                                level, totalN, batch+1, len(batchRandIxs), epoch+1, num_epochs, l, er))
                            batchErrors[batch] = er * len(batchSeqLengths)
                        elif mode == 'test':
                            l, pre, y = sess.run([model.loss, model.predictions, model.targetY], feed_dict=feedDict)
                            er = get_edit_distance([pre.values], [y.values], True, level)
                            print('\n{} mode, total:{},batch:{}/{},test loss={:.3f},mean test PER={:.3f}\n'.format(
                                level, totalN, batch+1, len(batchRandIxs), l, er))
                            batchErrors[batch] = er * len(batchSeqLengths)

                    # NOTE:
                    if er / batch_size == 1.0:
                        break

                    if batch % 30 == 0:
                        print('Truth:\n' + output_to_sequence(y, type=level))
                        print('Output:\n' + output_to_sequence(pre, type=level))

                    
                    if mode=='train' and ((epoch * len(batchRandIxs) + batch + 1) % 20 == 0 or (
                           epoch == num_epochs - 1 and batch == len(batchRandIxs) - 1)):
                        checkpoint_path = os.path.join(savedir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                        print('Model has been saved in {}'.format(savedir))
                end = time.time()
                delta_time = end - start
                print('Epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) + ' s')

                if mode=='train':
                    if (epoch + 1) % 1 == 0:
                        checkpoint_path = os.path.join(savedir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                        print('Model has been saved in {}'.format(savedir))
                    epochER = batchErrors.sum() / totalN
                    print('Epoch', epoch + 1, 'mean train error rate:', epochER)
                    logging(model, logfile, epochER, epoch, delta_time, mode='config')
                    logging(model, logfile, epochER, epoch, delta_time, mode=mode)


                if mode=='test':
                    with open(os.path.join(resultdir, level + '_result.txt'), 'a') as result:
                        result.write(output_to_sequence(y, type=level) + '\n')
                        result.write(output_to_sequence(pre, type=level) + '\n')
                        result.write('\n')
                    epochER = batchErrors.sum() / totalN
                    print(' test error rate:', epochER)
                    logging(model, logfile, epochER, mode=mode)



if __name__ == '__main__':
  runner = Runner()
  runner.run()
