#!/usr/bin/python
# -*- coding:utf-8 -*-
''' dynamic bidirectional rnn model for automatic speech recognition implemented in Tensorflow
author:

      iiiiiiiiiiii            iiiiiiiiiiii         !!!!!!!             !!!!!!    
      #        ###            #        ###           ###        I#        #:     
      #      ###              #      I##;             ##;       ##       ##      
            ###                     ###               !##      ####      #       
           ###                     ###                 ###    ## ###    #'       
         !##;                    `##%                   ##;  ##   ###  ##        
        ###                     ###                     $## `#     ##  #         
       ###        #            ###        #              ####      ####;         
     `###        -#           ###        `#               ###       ###          
     ##############          ##############               `#         #     
     
date:2016-12-07
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import datetime
import os
from six.moves import cPickle
from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

from utils.utils import load_batched_data
from utils.utils import describe
from utils.utils import setAttrs
from utils.utils import list_to_sparse_tensor
from utils.utils import dropout
from utils.utils import get_edit_distance

from utils.lnRNNCell import BasicRNNCell as lnBasicRNNCell
from utils.lnRNNCell import GRUCell as lnGRUCell
from utils.lnRNNCell import BasicLSTMCell as lnBasicLSTMCell

def build_multi_dynamic_brnn(args,
                             maxTimeSteps,
                             inputX,
                             cell_fn,
                             seqLengths,
                             time_major=True):
    hid_input = inputX
    for i in range(args.num_layer):
        scope = 'DBRNN_' + str(i + 1)
        forward_cell = cell_fn(args.num_hidden, activation=args.activation)
        backward_cell = cell_fn(args.num_hidden, activation=args.activation)
        # tensor of shape: [max_time, batch_size, input_size]
        outputs, output_states = bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                           inputs=hid_input,
                                                           dtype=tf.float32,
                                                           sequence_length=seqLengths,
                                                           time_major=True,
                                                           scope=scope)
        # forward output, backward ouput
        # tensor of shape: [max_time, batch_size, input_size]
        output_fw, output_bw = outputs
        # forward states, backward states
        output_state_fw, output_state_bw = output_states
        # output_fb = tf.concat(2, [output_fw, output_bw])
        output_fb = tf.concat([output_fw, output_bw], 2)
        shape = output_fb.get_shape().as_list()
        output_fb = tf.reshape(output_fb, [shape[0], shape[1], 2, int(shape[2] / 2)])
        hidden = tf.reduce_sum(output_fb, 2)
        hidden = dropout(hidden, args.keep_prob, (args.mode == 'train'))

        if i != args.num_layer - 1:
            hid_input = hidden
        else:
            outputXrs = tf.reshape(hidden, [-1, args.num_hidden])
            # output_list = tf.split(0, maxTimeSteps, outputXrs)
            output_list = tf.split(outputXrs, maxTimeSteps, 0)
            fbHrs = [tf.reshape(t, [args.batch_size, args.num_hidden]) for t in output_list]
    return fbHrs


class DBiRNN(object):
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
                self.cell_fn = rnn_cell.BasicRNNCell
            elif args.rnncell == 'gru':
                self.cell_fn = tf.contrib.rnn.GRUCell
            elif args.rnncell == 'lstm':
                self.cell_fn = core_rnn_cell_impl.BasicLSTMCell
            else:
                raise Exception("rnncell type not supported: {}".format(args.rnncell))

        self.build_graph(args, maxTimeSteps)

    @describe
    def build_graph(self, args, maxTimeSteps):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputX = tf.placeholder(tf.float32,
                                         shape=(maxTimeSteps, args.batch_size, args.num_feature))  # [maxL,32,39]
            inputXrs = tf.reshape(self.inputX, [-1, args.num_feature])
            # self.inputList = tf.split(0, maxTimeSteps, inputXrs) #convert inputXrs from [32*maxL,39] to [32,maxL,39]
            self.inputList = tf.split(inputXrs, maxTimeSteps, 0)  # convert inputXrs from [32*maxL,39] to [32,maxL,39]
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

            fbHrs = build_multi_dynamic_brnn(self.args, maxTimeSteps, self.inputX, self.cell_fn, self.seqLengths)
            with tf.name_scope('fc-layer'):
                with tf.variable_scope('fc'):
                    weightsClasses = tf.Variable(
                        tf.truncated_normal([args.num_hidden, args.num_class], name='weightsClasses'))
                    biasesClasses = tf.Variable(tf.zeros([args.num_class]), name='biasesClasses')
                    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in fbHrs]
            logits3d = tf.stack(logits)
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targetY, logits3d, self.seqLengths))
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
                tf.nn.ctc_beam_search_decoder(logits3d, self.seqLengths, merge_repeated=False)[0][0])
            if args.level == 'cha':
                self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))
            self.initial_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
