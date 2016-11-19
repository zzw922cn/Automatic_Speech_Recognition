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
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

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

	    if args.model == 'RNN':
		self.config = { 'name':args.model,
				'num_layer':args.num_layer,
				'num_hidden':args.num_hidden,
				'num_class':args.num_class,
				'activation':args.activation,
				'optimizer':args.optimizer,
				'learning rate':args.learning_rate
		}
		print(time.time())
                rnncell = rnn_cell.BasicRNNCell(args.num_hidden,activation=args.activation)
		print(time.time())
	        rnncells = rnn_cell.MultiRNNCell([rnncell]*args.num_layer)
  	        cell_out = tf.nn.rnn_cell.OutputProjectionWrapper(rnncells,args.num_class)
		print(time.time())
	        drnn, _ = tf.nn.dynamic_rnn(cell_out, self.inputX, dtype=tf.float32, scope = 'RNN')
    	        self.loss = tf.reduce_mean(ctc.ctc_loss(drnn, self.targetY, self.seqLengths))
    	        self.optimizer = args.optimizer(args.learning_rate).minimize(self.loss)
    	        self.logitsMaxTest = tf.slice(tf.argmax(drnn, 2), [0, 0], [self.seqLengths[0], 1])
    	        self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(drnn, self.seqLengths)[0][0])
    	        self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=False))/tf.to_float(tf.size(self.targetY.values))

		'''
		# there are some bugs when applying dynamic bidirectional rnn!!!
		# 2016-11-15
                f_RNNcell = rnn_cell.BasicRNNCell(args.num_hidden,activation=tf.nn.elu)
                b_RNNcell = rnn_cell.BasicRNNCell(args.num_hidden,activation=tf.nn.elu)
		fbRNN, _ = bidirectional_dynamic_rnn(f_RNNcell,b_RNNcell,inputs=self.inputX, dtype=tf.float32, scope='BiRNN')
		fbRNN1 = tf.concat(2,fbRNN)
    	        weightsOutH1 = tf.Variable(tf.truncated_normal([2, args.num_hidden],name='weightsOutH1'))
    	        biasesOutH1 = tf.Variable(tf.zeros([args.num_hidden]),name='biasesOutH1')
    	        weightsClasses = tf.Variable(tf.truncated_normal([args.num_hidden, args.num_class],name='weightsClasses'))
                biasesClasses = tf.Variable(tf.zeros([args.num_class]),name='biasesClasses')
    	        outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbRNN1]
    	        logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]
    	        logits3d = tf.pack(logits)
    	        self.loss = tf.reduce_mean(ctc.ctc_loss(logits3d, self.targetY, self.seqLengths))
    	        self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
    	        self.logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [self.seqLengths[0], 1])
    	        self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, self.seqLengths)[0][0])
    	        self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=False))/tf.to_float(tf.size(self.targetY.values))
		
	        '''
	    elif args.model == 'BiRNN':
		self.config = { 'name':args.model,
				'num_layer':args.num_layer,
				'num_hidden':args.num_hidden,
				'num_class':args.num_class,
				'activation':args.activation,
				'optimizer':args.optimizer,
				'learning rate':args.learning_rate
		}
    	        weightsOutH1 = tf.Variable(tf.truncated_normal([2, args.num_hidden],name='weightsOutH1'))
    	        biasesOutH1 = tf.Variable(tf.zeros([args.num_hidden]),name='biasesOutH1')
    	        weightsClasses = tf.Variable(tf.truncated_normal([args.num_hidden, args.num_class],name='weightsClasses'))
                biasesClasses = tf.Variable(tf.zeros([args.num_class]),name='biasesClasses')

                forwardH1 = rnn_cell.BasicRNNCell(args.num_hidden,activation=tf.nn.relu)
                backwardH1 = rnn_cell.BasicRNNCell(args.num_hidden,activation=tf.nn.relu)
                fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, self.inputList, dtype=tf.float32,scope='BDRNN_H1')
    	        fbH1rs = [tf.reshape(t, [args.batch_size, 2, args.num_hidden]) for t in fbH1]
    	        outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]
    	        logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]
    	        logits3d = tf.pack(logits)
    	        self.loss = tf.reduce_mean(ctc.ctc_loss(logits3d, self.targetY, self.seqLengths))
    	        self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)

    	        self.logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [self.seqLengths[0], 1])
    	        self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, self.seqLengths)[0][0])
    	        self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=False))/tf.to_float(tf.size(self.targetY.values))

	    elif args.model == 'CNN':
		self.config = { 'name':args.model,
				'num_layer':args.num_layer,
				'num_hidden':args.num_hidden,
				'num_class':args.num_class,
				'activation':args.activation,
				'optimizer':args.optimizer,
				'learning rate':args.learning_rate
		}
		filter1 = tf.Variable(tf.truncated_normal([3,3,1,128],dtype=tf.float32,name='filter1'))
		filter2 = tf.Variable(tf.truncated_normal([3,3,128,16],dtype=tf.float32,name='filter2'))

		conv_input = tf.reshape(self.inputX,[args.batch_size,maxTimeSteps,args.num_feature,1])
		conv1 = tf.nn.conv2d(conv_input,filter1,strides=[1,1,1,1],padding='SAME')
		conv1 = tf.nn.conv2d(conv1,filter2,strides=[1,1,1,1],padding='SAME')
		shape1 = conv1.get_shape().as_list()
		print(shape1)
		conv_output1 = tf.reshape(conv1,[shape1[1]*shape1[0],-1]) #seq,batch,feature
		conv_output1_list = tf.split(0,maxTimeSteps,conv_output1)

    	        weightsClasses = tf.Variable(tf.truncated_normal([shape1[2]*shape1[3], args.num_class],name='weightsClasses'))
                biasesClasses = tf.Variable(tf.zeros([args.num_class]),name='biasesClasses')

	  		
		
    	        logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in conv_output1_list]
    	        conv_output = tf.pack(logits)

    	        self.loss = tf.reduce_mean(ctc.ctc_loss(conv_output, self.targetY, self.seqLengths))
    	        self.optimizer = args.optimizer(args.learning_rate).minimize(self.loss)
    	        self.logitsMaxTest = tf.slice(tf.argmax(conv_output, 2), [0, 0], [self.seqLengths[0], 1])
    	        self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(conv_output, self.seqLengths)[0][0])
    	        self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=False))/tf.to_float(tf.size(self.targetY.values))

    	    self.initial_op = tf.initialize_all_variables()
	    self.saver = tf.train.Saver(tf.all_variables(),max_to_keep=5,keep_checkpoint_every_n_hours=1)
	    self.logfile = args.log_dir+str(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')+'.txt').replace(' ','').replace('/','')
	    self.var_op = tf.all_variables()
	    self.var_trainable_op = tf.trainable_variables()
