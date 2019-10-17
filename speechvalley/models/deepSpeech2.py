#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : deepSpeech2.py
# Author            : zewangzhang <zzw922cn@gmail.com>
# Date              : 17.10.2019
# Last Modified Date: 17.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : deepSpeech2.py
# Description  : Deep Speech2 model for Automatic Speech Recognition
# ******************************************************

import argparse
import time
import datetime
import os
from six.moves import cPickle
from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from speechvalley.utils import load_batched_data, describe, setAttrs, list_to_sparse_tensor, dropout, get_edit_distance
from speechvalley.utils import lnBasicRNNCell, lnGRUCell, lnBasicLSTMCell

def build_deepSpeech2(args,
                      maxTimeSteps,
                      inputX,
                      cell_fn,
                      seqLengths,
                      time_major=True):
    ''' Parameters:

          maxTimeSteps: maximum time steps of input spectrogram power
          inputX: spectrogram power of audios, [batch, freq_bin, time_len, in_channels]
          seqLengths: lengths of samples in a mini-batch
    '''

    # 3 2-D convolution layers
    layer1_filter = tf.get_variable('layer1_filter', shape=(41, 11, 1, 32), dtype=tf.float32)
    layer1_stride = [1, 2, 2, 1]
    layer2_filter = tf.get_variable('layer2_filter', shape=(21, 11, 32, 32), dtype=tf.float32)
    layer2_stride = [1, 2, 1, 1]
    layer3_filter = tf.get_variable('layer3_filter', shape=(21, 11, 32, 96), dtype=tf.float32)
    layer3_stride = [1, 2, 1, 1]
    layer1 = tf.nn.conv2d(inputX, layer1_filter, layer1_stride, padding='SAME')
    layer1 = tf.layers.batch_normalization(layer1, training=args.is_training)
    layer1 = tf.contrib.layers.dropout(layer1, keep_prob=args.keep_prob[0], is_training=args.is_training)

    layer2 = tf.nn.conv2d(layer1, layer2_filter, layer2_stride, padding='SAME')
    layer2 = tf.layers.batch_normalization(layer2, training=args.isTraining)
    layer2 = tf.contrib.layers.dropout(layer2, keep_prob=args.keep_prob[1], is_training=args.is_training)

    layer3 = tf.nn.conv2d(layer2, layer3_filter, layer3_stride, padding='SAME')
    layer3 = tf.layers.batch_normalization(layer3, training=args.isTraining)
    layer3 = tf.contrib.layers.dropout(layer3, keep_prob=args.keep_prob[2], is_training=args.is_training)

    # 4 recurrent layers
    # inputs must be [max_time, batch_size ,...]
    layer4_cell = cell_fn(args.num_hidden, activation=args.activation)
    layer4 = tf.nn.dynamic_rnn(layer4_cell, layer3, sequence_length=seqLengths, time_major=True)
    layer4 = tf.layers.batch_normalization(layer4, training=args.isTraining)
    layer4 = tf.contrib.layers.dropout(layer4, keep_prob=args.keep_prob[3], is_training=args.is_training)

    layer5_cell = cell_fn(args.num_hidden, activation=args.activation)
    layer5 = tf.nn.dynamic_rnn(layer5_cell, layer4, sequence_length=seqLengths, time_major=True)
    layer5 = tf.layers.batch_normalization(layer5, training=args.isTraining)
    layer5 = tf.contrib.layers.dropout(layer5, keep_prob=args.keep_prob[4], is_training=args.is_training)

    layer6_cell = cell_fn(args.num_hidden, activation=args.activation)
    layer6 = tf.nn.dynamic_rnn(layer6_cell, layer5, sequence_length=seqLengths, time_major=True)
    layer6 = tf.layers.batch_normalization(layer6, training=args.isTraining)
    layer6 = tf.contrib.layers.dropout(layer6, keep_prob=args.keep_prob[5], is_training=args.is_training)

    layer7_cell = cell_fn(args.num_hidden, activation=args.activation)
    layer7 = tf.nn.dynamic_rnn(layer7_cell, layer6, sequence_length=seqLengths, time_major=True)
    layer7 = tf.layers.batch_normalization(layer7, training=args.isTraining)
    layer7 = tf.contrib.layers.dropout(layer7, keep_prob=args.keep_prob[6], is_training=args.is_training)

    # fully-connected layer
    layer_fc = tf.layers.dense(layer7, args.num_hidden_fc)

    return layer_fc


class DeepSpeech2(object):
    def __init__(self, args, maxTimeSteps):
        self.args = args
        self.maxTimeSteps = maxTimeSteps
        if args.layerNormalization is True:
            if args.rnncell == 'rnn':
                self.cell_fn = lnBasicRNNCell
            elif args.rnncell == 'gru':
                self.cell_fn = lnGRUCell
            elif args.rnncell == 'lstm':
                self.cell_fn = lnBasicLSTMCell
            else:
                raise Exception("rnncell type not supported: {}".format(args.rnncell))
        else:
            if args.rnncell == 'rnn':
                self.cell_fn = tf.contrib.rnn.BasicRNNCell
            elif args.rnncell == 'gru':
                self.cell_fn = tf.contrib.rnn.GRUCell
            elif args.rnncell == 'lstm':
                self.cell_fn = tf.contrib.rnn.BasicLSTMCell
            else:
                raise Exception("rnncell type not supported: {}".format(args.rnncell))
        self.build_graph(args, maxTimeSteps)

    @describe
    def build_graph(self, args, maxTimeSteps):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # according to DeepSpeech2 paper, input is the spectrogram power of audio, but if you like,
            # you can also use mfcc feature as the input.
            self.inputX = tf.placeholder(tf.float32,
                                         shape=(maxTimeSteps, args.batch_size, args.num_feature))
            inputXrs = tf.reshape(self.inputX, [args.batch_size, args.num_feature, maxTimeSteps, 1])
            #self.inputList = tf.split(inputXrs, maxTimeSteps, 0)  # convert inputXrs from [32*maxL,39] to [32,maxL,39]

            self.targetIxs = tf.placeholder(tf.int64)
            self.targetVals = tf.placeholder(tf.int32)
            self.targetShape = tf.placeholder(tf.int64)
            self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)
            self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size))
            self.config = {'name': args.model,
                           'rnncell': self.cell_fn,
                           'num_layer': args.num_layer,
                           'num_hidden': args.num_hidden,
                           'num_class': args.num_class,
                           'activation': args.activation,
                           'optimizer': args.optimizer,
                           'learning rate': args.learning_rate,
                           'keep prob': args.keep_prob,
                           'batch size': args.batch_size}

            output_fc = build_deepSpeech2(self.args, maxTimeSteps, self.inputX, self.cell_fn, self.seqLengths)
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targetY, output_fc, self.seqLengths))
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()

            if args.grad_clip == -1:
                # not apply gradient clipping
                self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
            else:
                # apply gradient clipping
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.var_trainable_op), args.grad_clip)
                opti = tf.train.AdamOptimizer(args.learning_rate)
                self.optimizer = opti.apply_gradients(zip(grads, self.var_trainable_op))
            self.predictions = tf.to_int32(
                tf.nn.ctc_beam_search_decoder(output_fc, self.seqLengths, merge_repeated=False)[0][0])
            if args.level == 'cha':
                self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))
            self.initial_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
