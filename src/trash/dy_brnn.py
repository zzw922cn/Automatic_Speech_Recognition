#-*- coding:utf-8 -*-
#!/usr/bin/python
''' bidirectional rnn model for automatic speech recognition implemented in Tensorflow
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
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

from utils import load_batched_data
from utils import describe
from utils import setAttrs
from utils import build_weight
from utils import build_forward_layer
from utils import build_conv_layer

class BiRNN(object):
    def __init__(self,args,maxTimeSteps):
	self.args = args
	self.maxTimeSteps = maxTimeSteps
	if args.rnncell == 'rnn':
            self.cell_fn = rnn_cell.BasicRNNCell
        elif args.rnncell == 'gru':
            self.cell_fn = rnn_cell.GRUCell
        elif args.rnncell == 'lstm':
            self.cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))
	self.build_graph(args,maxTimeSteps)

    @describe
    def build_graph(self, args, maxTimeSteps):
	self.graph = tf.Graph()
	with self.graph.as_default():
    	    self.inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, args.batch_size, args.num_feature)) #[maxL,32,39]
	    self.inputXX = tf.reshape(self.inputX,shape=(args.batch_size,maxTimeSteps,args.num_feature))
    	    inputXrs = tf.reshape(self.inputX, [-1, args.num_feature])
    	    #self.inputList = tf.split(0, maxTimeSteps, inputXrs) #convert inputXrs from [32*maxL,39] to [32,maxL,39]
    	    #self.inputnew = tf.reshape(self.inputX, [1, 0, 2])
            self.targetIxs = tf.placeholder(tf.int64)
    	    self.targetVals = tf.placeholder(tf.int32)
    	    self.targetShape = tf.placeholder(tf.int64)
    	    self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)
            self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size))
	    self.config = { 'name':args.model,
			    'rnncell':self.cell_fn,
			    'num_layer':args.num_layer,
			    'num_hidden':args.num_hidden,
			    'num_class':args.num_class,
			    'activation':args.activation,
			    'optimizer':args.optimizer,
			    'learning rate':args.learning_rate
	    }	    

	    # forward layer
            forwardH1 = self.cell_fn(args.num_hidden,activation=tf.nn.relu)
	    # backward layer
            backwardH1 = self.cell_fn(args.num_hidden,activation=tf.nn.relu)
	    # bi-directional layer
            fbH1, state = bidirectional_dynamic_rnn(forwardH1, backwardH1, self.inputXX, sequence_length=self.seqLengths, dtype=tf.float32, scope='BDRNN_H1')
	    fbH1 = tf.concat(2, fbH1)
	    print(fbH1.get_shape)
            shape = fbH1.get_shape().as_list()
	    fbH1 = tf.reshape(fbH1,[shape[0]*shape[1],-1]) #seq*batch,feature
	    fbH1_list = tf.split(0,shape[1],fbH1)
    	    logits = [build_forward_layer(t,[shape[2],args.num_class],kernel='linear') for t in fbH1_list]
    	    logits3d = tf.pack(logits)
    	    self.loss = tf.reduce_mean(ctc.ctc_loss(logits3d, self.targetY, self.seqLengths))
    	    self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
    	    self.logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [self.seqLengths[0], 1])
    	    self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, self.seqLengths)[0][0])
    	    self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=False))/tf.to_float(tf.size(self.targetY.values))
	    self.initial_op = tf.initialize_all_variables()
	    self.saver = tf.train.Saver(tf.all_variables(),max_to_keep=5,keep_checkpoint_every_n_hours=1)
	    self.logfile = args.log_dir+str(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')+'.txt').replace(' ','').replace('/','')
	    self.var_op = tf.all_variables()
	    self.var_trainable_op = tf.trainable_variables()
