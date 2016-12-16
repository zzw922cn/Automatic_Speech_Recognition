#-*- coding:utf-8 -*-
#!/usr/bin/python
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

Models for automatic speech recognition is as follows:
	RNN	--
	BiRNN	--
	CNN	--
	Res-CNN	--
	
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
                rnncell = rnn_cell.BasicRNNCell(args.num_hidden,activation=args.activation)
	        rnncells = rnn_cell.MultiRNNCell([rnncell]*args.num_layer)
  	        cell_out = tf.nn.rnn_cell.OutputProjectionWrapper(rnncells,args.num_class)

	        drnn, _ = tf.nn.dynamic_rnn(cell_out, self.inputX, dtype=tf.float32, scope = 'RNN')
    	        self.loss = tf.reduce_mean(ctc.ctc_loss(drnn, self.targetY, self.seqLengths))
    	        self.optimizer = args.optimizer(args.learning_rate).minimize(self.loss)
    	        self.logitsMaxTest = tf.slice(tf.argmax(drnn, 2), [0, 0], [self.seqLengths[0], 1])
    	        self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(drnn, self.seqLengths)[0][0])
    	        self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))/tf.to_float(tf.size(self.targetY.values))

	    elif args.model == 'Res-RNN':
		self.config = { 'name':args.model,
				'num_layer':args.num_layer,
				'num_hidden':args.num_hidden,
				'num_class':args.num_class,
				'activation':args.activation,
				'optimizer':args.optimizer,
				'learning rate':args.learning_rate
		}
                rnncell1 = rnn_cell.BasicRNNCell(args.num_hidden,activation=args.activation)

                rnncell2 = rnn_cell.BasicRNNCell(args.num_hidden,activation=args.activation)
                rnncell3 = rnn_cell.BasicRNNCell(args.num_hidden,activation=args.activation)
                rnncell4 = rnn_cell.BasicRNNCell(args.num_hidden,activation=args.activation)

                rnncell5 = rnn_cell.BasicRNNCell(args.num_hidden,activation=args.activation)
                rnncell6 = rnn_cell.BasicRNNCell(args.num_hidden,activation=args.activation)
                rnncell7 = rnn_cell.BasicRNNCell(args.num_hidden,activation=args.activation)

	        block1 = rnn_cell.MultiRNNCell([rnncell2,rnncell3,rnncell4])
	        block2 = rnn_cell.MultiRNNCell([rnncell5,rnncell6,rnncell7])

	        node1 = rnn_cell.MultiRNNCell([rnncell1,block1])
	        #node1_sum = rnn_cell.MultiRNNCell([rnncell1,node1])
		#node1_sum = node1
		node1_sum = rnncell1 + node1
	        node2 = rnn_cell.MultiRNNCell([node1_sum,block2])
	        node2_sum = rnn_cell.MultiRNNCell([node1_sum,node2])

		node = node2_sum
  	        cell_out1 = tf.nn.rnn_cell.OutputProjectionWrapper(node,args.num_class)
	        drnn1, _ = tf.nn.dynamic_rnn(cell_out1, self.inputX, dtype=tf.float32, scope = 'RNN')

		end = drnn1
    	        self.loss = tf.reduce_mean(ctc.ctc_loss(end, self.targetY, self.seqLengths))
    	        self.optimizer = args.optimizer(args.learning_rate).minimize(self.loss)
    	        self.logitsMaxTest = tf.slice(tf.argmax(end, 2), [0, 0], [self.seqLengths[0], 1])
    	        self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(end, self.seqLengths)[0][0])
    	        self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=False))/tf.to_float(tf.size(self.targetY.values))

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
    	        self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))/tf.to_float(tf.size(self.targetY.values))

	    elif args.model == 'CNN':
		self.config = { 'name':args.model,
				'num_layer':args.num_layer,
				'num_hidden':args.num_hidden,
				'num_class':args.num_class,
				'activation':args.activation,
				'optimizer':args.optimizer,
				'learning rate':args.learning_rate
		}
		filter1 = tf.Variable(tf.truncated_normal([3,3,1,16],dtype=tf.float32,name='filter1'))
		filter2 = tf.Variable(tf.truncated_normal([3,3,16,16],dtype=tf.float32,name='filter2'))
		filter3 = tf.Variable(tf.truncated_normal([3,3,16,16],dtype=tf.float32,name='filter3'))
		filter4 = tf.Variable(tf.truncated_normal([3,3,16,16],dtype=tf.float32,name='filter4'))
		filter5 = tf.Variable(tf.truncated_normal([3,3,16,16],dtype=tf.float32,name='filter5'))
		filter6 = tf.Variable(tf.truncated_normal([3,3,16,16],dtype=tf.float32,name='filter6'))
		filter7 = tf.Variable(tf.truncated_normal([3,3,16,16],dtype=tf.float32,name='filter7'))
		filter8 = tf.Variable(tf.truncated_normal([3,3,16,16],dtype=tf.float32,name='filter8'))
		filter9 = tf.Variable(tf.truncated_normal([3,3,16,16],dtype=tf.float32,name='filter9'))
		filter10 = tf.Variable(tf.truncated_normal([3,3,16,16],dtype=tf.float32,name='filter10'))
		filter11 = tf.Variable(tf.truncated_normal([3,3,16,16],dtype=tf.float32,name='filter11'))
		filter12 = tf.Variable(tf.truncated_normal([3,3,16,1],dtype=tf.float32,name='filter12'))

		conv_input = tf.reshape(self.inputX,[args.batch_size,maxTimeSteps,args.num_feature,1])
		'''
		conv1 = tf.nn.conv2d(conv_input,filter1,strides=[1,1,1,1],padding='SAME')
		conv2 = tf.nn.conv2d(conv1,filter2,strides=[1,1,1,1],padding='SAME')
		conv3 = tf.nn.conv2d(conv2,filter3,strides=[1,1,1,1],padding='SAME')
		conv4 = tf.nn.conv2d(conv3,filter4,strides=[1,1,1,1],padding='SAME')
		conv5 = tf.nn.conv2d(conv4,filter5,strides=[1,1,1,1],padding='SAME')
		conv6 = tf.nn.conv2d(conv5,filter6,strides=[1,1,1,1],padding='SAME')
		conv7 = tf.nn.conv2d(conv6,filter7,strides=[1,1,1,1],padding='SAME')
		conv8 = tf.nn.conv2d(conv7,filter8,strides=[1,1,1,1],padding='SAME')
		conv9 = tf.nn.conv2d(conv8,filter9,strides=[1,1,1,1],padding='SAME')
		conv10 = tf.nn.conv2d(conv9,filter10,strides=[1,1,1,1],padding='SAME')
		conv11 = tf.nn.conv2d(conv10,filter11,strides=[1,1,1,1],padding='SAME')
		conv12 = tf.nn.conv2d(conv11,filter12,strides=[1,1,1,1],padding='SAME')
		'''
		end = conv_input
		shape1 = end.get_shape().as_list()
		conv_output1 = tf.reshape(end,[shape1[1]*shape1[0],-1]) #seq*batch,feature
		conv_output1_list = tf.split(0,maxTimeSteps,conv_output1)

    	        weightsClasses1 = tf.Variable(tf.truncated_normal([shape1[2]*shape1[3], 512],name='weightsClasses1'))
                biasesClasses1 = tf.Variable(tf.zeros([512]),name='biasesClasses1')
    	        weightsClasses2 = tf.Variable(tf.truncated_normal([512, args.num_class],name='weightsClasses2'))
                biasesClasses2 = tf.Variable(tf.zeros([args.num_class]),name='biasesClasses2')

    	        logits = [tf.matmul(t, weightsClasses1) + biasesClasses1 for t in conv_output1_list]
    	        logits = [tf.matmul(t, weightsClasses2) + biasesClasses2 for t in logits]
    	        conv_output = tf.pack(logits)

    	        self.loss = tf.reduce_mean(ctc.ctc_loss(conv_output, self.targetY, self.seqLengths))
    	        self.optimizer = args.optimizer(args.learning_rate).minimize(self.loss)
    	        self.logitsMaxTest = tf.slice(tf.argmax(conv_output, 2), [0, 0], [self.seqLengths[0], 1])
    	        self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(conv_output, self.seqLengths)[0][0])
    	        self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))/tf.to_float(tf.size(self.targetY.values))

    	    self.initial_op = tf.initialize_all_variables()
	    self.saver = tf.train.Saver(tf.all_variables(),max_to_keep=5,keep_checkpoint_every_n_hours=1)
	    self.logfile = args.log_dir+str(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')+'.txt').replace(' ','').replace('/','')
	    self.var_op = tf.all_variables()
	    self.var_trainable_op = tf.trainable_variables()
