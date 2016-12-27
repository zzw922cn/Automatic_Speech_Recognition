#-*- coding:utf-8 -*-
#!/usr/bin/python
''' automatic speech recognition implemented in Tensorflow
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

from src.utils.utils import load_batched_data
from src.utils.utils import describe
from src.utils.utils import setAttrs
from src.utils.utils import build_weight
from src.utils.utils import build_forward_layer
from src.utils.utils import build_conv_layer
   


class Seq2Seq(object):
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
	    #self.inputXX = tf.reshape(self.inputX,shape=(args.batch_size,maxTimeSteps,args.num_feature))
    	    inputXrs = tf.reshape(self.inputX, [-1, args.num_feature])
    	    self.inputList = tf.split(0, maxTimeSteps, inputXrs) #convert inputXrs from [32*maxL,39] to [32,maxL,39]
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
