#-*- coding:utf-8 -*-
#!/usr/bin/python
''' deep residual model for automatic speech recognition implemented in Tensorflow
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
from tensorflow.contrib.rnn.python.ops import rnn_cell
#from tensorflow.python.ops.rnn import bidirectional_rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
bidirectional_rnn = tf.contrib.rnn.static_bidirectional_rnn

from utils.utils import load_batched_data
from utils.utils import describe
from utils.utils import setAttrs
from utils.utils import build_weight
from utils.utils import build_forward_layer
from utils.utils import build_conv_layer


def build_residual_block(inpt, out_channels, down_sample=False, projection=False, name='block1'):
    ''' building a residual block,inside the block is [(3,3),(3,3)]
    '''
    inp_channels = inpt.get_shape().as_list()[3]
    if down_sample:
	filter_ = [1,2,2,1]
	inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    conv1 = build_conv_layer(inpt, [3, 3, inp_channels, out_channels], 1, name=name+'_conv1')
    conv2 = build_conv_layer(conv1, [3, 3, out_channels, out_channels], 1, name=name+'_conv2')

    if inp_channels != out_channels:
	if projection:
	    input_layer = build_conv_layer(inpt, [1, 1, inp_channels, out_channels], 2, name=name+'_input')
	else:
	    input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, out_channels - inp_channels]])
    else:
	input_layer = inpt
    res_block = conv2 + input_layer
    return res_block
    
def build_resnet(inpt,maxTimeSteps,depth,width,num_class):
    num_conv = int((depth-2)/8)
    layers = []
    with tf.variable_scope('conv1'):
	conv1 = build_conv_layer(inpt,[3,3,1,width],1,name='conv1')
	layers.append(conv1)
    for i in range(num_conv):
	with tf.variable_scope('conv2_%d'%(i+1)):
	    conv2_1 = build_residual_block(layers[-1],width,name='block1')
	    conv2_2 = build_residual_block(conv2_1,width,name='block2')
	    layers.append(conv2_1)
	    layers.append(conv2_2)
	assert conv2_2.get_shape().as_list()[3] == width

    for i in range(num_conv):
	with tf.variable_scope('conv3_%d'%(i+1)):
	    conv3_1 = build_residual_block(layers[-1],width,name='block1')
	    conv3_2 = build_residual_block(conv3_1,width,name='block2')
	    layers.append(conv3_1)
	    layers.append(conv3_2)
	assert conv3_2.get_shape().as_list()[3] == width

    with tf.variable_scope('fc'):
	global_pool = tf.reduce_mean(layers[-1], [3])
	print(global_pool.get_shape().as_list())
	shape = global_pool.get_shape().as_list()
	res_output = tf.reshape(global_pool,[shape[0]*shape[1],-1]) #seq*batch,feature
	#res_output_list = tf.split(0,maxTimeSteps,res_output)
	res_output_list = tf.split(res_output, maxTimeSteps, 0)
    	logits = [build_forward_layer(t, [shape[2],num_class], 
		kernel='linear', name_scope='out_proj_'+str(idx)) 
		for idx,t in enumerate(res_output_list)]
    	conv_output = tf.stack(logits)
    return conv_output

class ResNet(object):
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
    	    self.inputList = tf.split(inputXrs, maxTimeSteps, 0) #convert inputXrs from [32*maxL,39] to [32,maxL,39]

            self.targetIxs = tf.placeholder(tf.int64)
    	    self.targetVals = tf.placeholder(tf.int32)
    	    self.targetShape = tf.placeholder(tf.int64)
    	    self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)
            self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size))
	    depth = 10
	    width = 8
	    self.config = { 'name':'residual network',
			    'num_layer':depth,
			    'num_featuremap':width,
			    'num_class':args.num_class,
			    'optimizer':args.optimizer,
			    'learning rate':args.learning_rate
		}

	    inpt = tf.reshape(self.inputX,[args.batch_size,maxTimeSteps,args.num_feature,1])
            conv_output = build_resnet(inpt,maxTimeSteps,depth,width,args.num_class)
    	    self.loss = tf.reduce_mean(ctc.ctc_loss(self.targetY, conv_output, self.seqLengths))
    	    self.optimizer = args.optimizer(args.learning_rate).minimize(self.loss)
    	    self.logitsMaxTest = tf.slice(tf.argmax(conv_output, 2), [0, 0], [self.seqLengths[0], 1])
    	    self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(conv_output, self.seqLengths)[0][0])
    	    self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=False))/tf.to_float(tf.size(self.targetY.values))
	    self.initial_op = tf.global_variables_initializer()
	    self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=2,keep_checkpoint_every_n_hours=1)
	    self.var_op = tf.global_variables()
	    self.var_trainable_op = tf.trainable_variables()
