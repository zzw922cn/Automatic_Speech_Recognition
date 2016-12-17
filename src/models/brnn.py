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

from src.utils.utils import load_batched_data
from src.utils.utils import describe
from src.utils.utils import setAttrs
from src.utils.utils import build_weight
from src.utils.utils import build_forward_layer
from src.utils.utils import build_conv_layer

def build_multi_brnn(args,maxTimeSteps,inputList,cell_fn,seqLengths):
    hid_input = inputList
    for i in range(args.num_layer):
	scope = 'BRNN_'+str(i+1)

        forward_cell = cell_fn(args.num_hidden,activation=args.activation)
        backward_cell = cell_fn(args.num_hidden,activation=args.activation)
        fbH, f_state, b_state = bidirectional_rnn(forward_cell,backward_cell,
					hid_input,dtype=tf.float32,sequence_length=seqLengths,scope=scope)

	fbHrs = [tf.reshape(t, [args.batch_size, 2, args.num_hidden]) for t in fbH]
	if i != args.num_layer-1:
            # output size is seqlength*batchsize*2*hidden
            output = tf.convert_to_tensor(fbHrs, dtype=tf.float32)

            # output size is seqlength*batchsize*hidden
	    output = tf.reduce_sum(output,2)

	    # outputXrs is of size [seqlenth*batchsize,num_hidden]
    	    outputXrs = tf.reshape(output, [-1, args.num_hidden])
    	    hid_input = tf.split(0, maxTimeSteps, outputXrs) #convert inputXrs from [32*maxL,39] to [32,maxL,39]

    return fbHrs
    


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

	    fbHrs = build_multi_brnn(self.args,maxTimeSteps,self.inputList,self.cell_fn,self.seqLengths)
	    with tf.name_scope('fc-layer'):
                with tf.variable_scope('fc'):
		    weightsOutH1 = tf.Variable(tf.truncated_normal([2, args.num_hidden],name='weightsOutH1'))
    	            biasesOutH1 = tf.Variable(tf.zeros([args.num_hidden]),name='biasesOutH1')
    	            weightsClasses = tf.Variable(tf.truncated_normal([args.num_hidden, args.num_class],name='weightsClasses'))
                    biasesClasses = tf.Variable(tf.zeros([args.num_class]),name='biasesClasses')
    	            outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbHrs]
    	            logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]
    	    logits3d = tf.pack(logits)
    	    self.loss = tf.reduce_mean(ctc.ctc_loss(logits3d, self.targetY, self.seqLengths))
	    #self.var_op = tf.all_variables()
	    self.var_op = tf.global_variables()
	    self.var_trainable_op = tf.trainable_variables()

	    if args.grad_clip == -1:
		# not apply gradient clipping
		self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
	    else:
		# apply gradient clipping
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.var_trainable_op),args.grad_clip)
                opti = tf.train.AdamOptimizer(args.learning_rate)
                self.optimizer = opti.apply_gradients(zip(grads, self.var_trainable_op))
    	    self.logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [self.seqLengths[0], 1])
    	    self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, self.seqLengths)[0][0])
    	    self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=False))/tf.to_float(tf.size(self.targetY.values))
	    #self.initial_op = tf.initialize_all_variables()
	    self.initial_op = tf.global_variables_initializer()
	    self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=5,keep_checkpoint_every_n_hours=1)
	    self.logfile = args.log_dir+str(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')+'.txt').replace(' ','').replace('/','')
