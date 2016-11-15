#-*- coding:utf-8 -*-
''' model for automatic speech recognition implemented in Tensorflow
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
     
date:2016-11-07
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
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn

from utils import load_batched_data
from utils import describe
from utils import setAttrs


class Model(object):

    def __init__(self,args,maxTimeSteps):
	self.args = args
	self.maxTimeSteps = maxTimeSteps
	self.build_graph(args,maxTimeSteps)

    @describe
    def build_graph(self, args, maxTimeSteps):
	self.graph = tf.Graph()
	with self.graph.as_default():

    	    self.inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, args.batch_size, args.num_feature)) #[maxL,32,39]
    	    inputXrs = tf.reshape(self.inputX, [-1, args.num_feature])
    	    self.inputList = tf.split(0, maxTimeSteps, inputXrs) #convert inputXrs from [32*maxL,39] to [32,maxL,39]
            self.targetIxs = tf.placeholder(tf.int64)
    	    self.targetVals = tf.placeholder(tf.int32)
    	    self.targetShape = tf.placeholder(tf.int64)
    	    self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)
            self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size))

    	    weightsOutH1 = tf.Variable(tf.truncated_normal([2, args.num_hidden],
                                                   	stddev=np.sqrt(2.0 / (2*args.num_hidden))),name='weightsOutH1')
    	    biasesOutH1 = tf.Variable(tf.zeros([args.num_hidden]),name='biasesOutH1')
    	    weightsOutH2 = tf.Variable(tf.truncated_normal([2, args.num_hidden],
                                                   	stddev=np.sqrt(2.0 / (2*args.num_hidden))),name='weightsOutH2')
    	    biasesOutH2 = tf.Variable(tf.zeros([args.num_hidden]),name='biasesOutH2')
    	    weightsClasses = tf.Variable(tf.truncated_normal([args.num_hidden, args.num_class],
       	                                                stddev=np.sqrt(2.0 / args.num_hidden)),name='weightsClasses')
            biasesClasses = tf.Variable(tf.zeros([args.num_class]),name='biasesClasses')

            forwardH1 = rnn_cell.BasicRNNCell(args.num_hidden,activation=tf.nn.relu)
            backwardH1 = rnn_cell.BasicRNNCell(args.num_hidden,activation=tf.nn.relu)

	    # test for dynamic rnn
            testRNNcell = rnn_cell.BasicRNNCell(args.num_hidden,activation=tf.nn.elu)
  	    cell_out = tf.nn.rnn_cell.OutputProjectionWrapper(testRNNcell,args.num_class)
	    drnn, _ = tf.nn.dynamic_rnn(cell_out, self.inputX, dtype=tf.float32, scope = 'testRNN')
	    drnn_shape = tf.shape(drnn)
    	    self.loss = tf.reduce_mean(ctc.ctc_loss(drnn, self.targetY, self.seqLengths))
    	    self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
    	    self.logitsMaxTest = tf.slice(tf.argmax(drnn, 2), [0, 0], [self.seqLengths[0], 1])
    	    self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(drnn, self.seqLengths)[0][0])
    	    self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=False))/tf.to_float(tf.size(self.targetY.values))

	    '''
            fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, self.inputList, dtype=tf.float32,scope='BDRNN_H1')
    	    fbH1rs = [tf.reshape(t, [args.batch_size, 2, args.num_hidden]) for t in fbH1]
    	    outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

    	    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

    	    logits3d = tf.pack(logits)
    	    self.loss = tf.reduce_mean(ctc.ctc_loss(logits3d, self.targetY, self.seqLengths))
    	    self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)

    	    self.logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [self.seqLengths[0], 1])
    	    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, self.seqLengths)[0][0])
    	    self.errorRate = tf.reduce_sum(tf.edit_distance(predictions, self.targetY, normalize=False))/tf.to_float(tf.size(self.targetY.values))
	    '''
    	    self.initial_op = tf.initialize_all_variables()
	    self.saver = tf.train.Saver(tf.all_variables(),max_to_keep=5,keep_checkpoint_every_n_hours=1)
	    self.logfile = args.log_dir+str(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')+'.txt').replace(' ','').replace('/','')
	    self.var_op = tf.all_variables()
	    self.var_trainable_op = tf.trainable_variables()
