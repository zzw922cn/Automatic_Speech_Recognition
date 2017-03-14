# -*- coding:utf-8 -*-
#!/usr/bin/python

''' This file is designed to train the models of ASR
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
     
date:2016-11-09
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.dont_write_bytecode = True

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

bidirectional_rnn = tf.contrib.rnn.static_bidirectional_rnn
from utils.utils import load_batched_data
from utils.utils import describe
from utils.utils import getAttrs
from utils.utils import output_to_sequence
from utils.utils import list_dirs
from utils.utils import logging
from utils.utils import count_params
from utils.utils import target2phoneme
from utils.utils import get_edit_distance

from models.resnet import ResNet
from models.brnn import BiRNN
from models.dynamic_brnn import DBiRNN

class Trainer(object):
    
    def __init__(self):
	parser = argparse.ArgumentParser()
        parser.add_argument('--train_mfcc_dir', type=str, default='/home/pony/github/data/timit/train/mfcc/',
                       help='data directory containing mfcc numpy files, usually end with .npy')

        parser.add_argument('--train_label_dir', type=str, default='/home/pony/github/data/timit/train/label/',
                       help='data directory containing label numpy files, usually end with .npy')

        parser.add_argument('--test_mfcc_dir', type=str, default='/home/pony/github/data/timit/test/mfcc/',
                       help='data directory containing mfcc numpy files, usually end with .npy')

        parser.add_argument('--test_label_dir', type=str, default='/home/pony/github/data/timit/test/label/',
                       help='data directory containing label numpy files, usually end with .npy')

	parser.add_argument('--log_dir', type=str, default='/home/pony/github/data/ASR/log/',
                       help='directory to log events while training')

	parser.add_argument('--model', default='DBiRNN',
		       help='model for ASR:DBiRNN,BiRNN,ResNet,...')

	parser.add_argument('--keep_prob', type=float, default=0.6,
		       help='set the keep probability of layer for dropout')

	parser.add_argument('--rnncell', type=str, default='gru',
		       help='rnn cell, 3 choices:rnn,lstm,gru')

        parser.add_argument('--num_layer', type=int, default=2,
                       help='set the number of hidden layer or bidirectional layer')

        parser.add_argument('--activation', default=tf.nn.relu,
                       help='set the activation function of each layer')

        parser.add_argument('--optimizer', type=type, default=tf.train.AdamOptimizer,
                       help='set the optimizer to train the model,eg:AdamOptimizer,GradientDescentOptimizer')
	
        parser.add_argument('--grad_clip', default=5,
                       help='set gradient clipping when backpropagating errors')

	parser.add_argument('--keep', type=bool, default=False,
                       help='train the model based on model saved')

	parser.add_argument('--save', type=bool, default=True,
                       help='to save the model in the disk')

	parser.add_argument('--mode', type=str, default='train',
                       help='test the model based on trained parameters, but at present, we can"t test during training.')

        parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='set the step size of each iteration')

        parser.add_argument('--num_epoch', type=int, default=5,
                       help='set the total number of training epochs')

        parser.add_argument('--batch_size', type=int, default=32,
                       help='set the number of training samples in a mini-batch')

        parser.add_argument('--test_batch_size', type=int, default=512,
                       help='set the number of testing samples in a mini-batch')

        parser.add_argument('--num_feature', type=int, default=39,
                       help='set the dimension of feature, ie: 39 mfccs, you can set 39 ')

        parser.add_argument('--num_hidden', type=int, default=128,
                       help='set the number of neurons in hidden layer')

        parser.add_argument('--num_class', type=int, default=62,
                       help='set the number of labels in the output layer, if timit phonemes, it is 62; if timit characters, it is 28; if libri characters, it is 28')

        parser.add_argument('--save_dir', type=str, default='/home/pony/github/data/ASR/save/',
                       help='set the directory to save the model, containing checkpoint file and parameter file')

        parser.add_argument('--model_checkpoint_path', type=str, default='/home/pony/github/data/ASR/save/',
                       help='set the directory to restore the model, containing checkpoint file and parameter file')

	self.args = parser.parse_args()
	self.logfile = self.args.log_dir+str(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')+'.txt').replace(' ','').replace('/','')

    @describe
    def load_data(self,args,mode='train'):
  	if mode == 'train':	
            return load_batched_data(args.train_mfcc_dir,args.train_label_dir,args.batch_size)
  	elif mode == 'test':	
	    args.batch_size = args.test_batch_size
            return load_batched_data(args.test_mfcc_dir,args.test_label_dir,args.test_batch_size)
	else:
	    raise TypeError('mode should be train or test.')

    def train(self):
	# load data
	args = self.args
        batchedData, maxTimeSteps, totalN = self.load_data(args,mode='train')

	# put the maxTimeSteps into network,
	# but it's not efficient, can we make
	# the dynamic network?
	if args.model == 'ResNet':
	    model = ResNet(args,maxTimeSteps)
	elif args.model == 'BiRNN':
	    model = BiRNN(args,maxTimeSteps)
	elif args.model == 'DBiRNN':
	    model = DBiRNN(args,maxTimeSteps)
	else:
	    pass

	# count the num of params
	num_params = count_params(model,mode='trainable')
	all_num_params = count_params(model,mode='all')
	model.config['trainable params'] = num_params
	model.config['all params'] = all_num_params
	print(model.config)

        with tf.Session(graph=model.graph) as sess:
	    # restore from stored model
	    if args.keep == True:
		ckpt = tf.train.get_checkpoint_state(args.save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
    		    print('Model restored from:'+args.save_dir) 
	    else:
    	        print('Initializing')
    	        sess.run(model.initial_op)
    
    	    for epoch in range(args.num_epoch):
		## training
		start = time.time()
        	print('Epoch', epoch+1, '...')
        	batchErrors = np.zeros(len(batchedData))
        	batchRandIxs = np.random.permutation(len(batchedData)) 
        	for batch, batchOrigI in enumerate(batchRandIxs):
            	    batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                    batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                    feedDict = {model.inputX: batchInputs, model.targetIxs: batchTargetIxs, model.targetVals: batchTargetVals,model.targetShape: batchTargetShape, model.seqLengths: batchSeqLengths}

                    _, l, er, lmt, pre, y = sess.run([model.optimizer, model.loss, model.errorRate, model.logitsMaxTest, model.predictions, model.targetY], feed_dict=feedDict)

		    er = get_edit_distance([pre.values],[y.values], mode='train')

                    
                    print('Epoch', epoch+1, 'Minibatch', batch+1, 'loss:', l)
                    print('Epoch', epoch+1, 'Minibatch', batch+1, 'error rate:', er)
            	    batchErrors[batch] = er*len(batchSeqLengths)
		    
		    if (args.save==True) and  ((epoch*len(batchRandIxs)+batch+1)%5000==0 or (epoch==args.num_epoch-1 and batch==len(batchRandIxs)-1)):
		        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        model.saver.save(sess,checkpoint_path,global_step=epoch)
                        print('Model has been saved in file')
	        end = time.time()
		delta_time = end-start
		print('Epoch '+str(epoch+1)+' needs time:'+str(delta_time)+' s')	

		if args.save==True and (epoch+1)%1==0:
		    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    model.saver.save(sess,checkpoint_path,global_step=epoch)
                    print('Model has been saved in file')
                epochER = batchErrors.sum() / totalN
                print('Epoch', epoch+1, 'train error rate:', epochER)
	        logging(model,self.logfile,epochER,epoch,delta_time,mode='config')
		logging(model,self.logfile,epochER,epoch,delta_time,mode='train')
	
    def test(self):
	# load data
	args = self.args
        batchedData, maxTimeSteps, totalN = self.load_data(args,mode='test')
	if args.model == 'ResNet':
	    model = ResNet(args,maxTimeSteps)
	elif args.model == 'BiRNN':
	    model = BiRNN(args,maxTimeSteps)
	elif args.model == 'DBiRNN':
	    model = DBiRNN(args,maxTimeSteps)

	num_params = count_params(model,mode='trainable')
	all_num_params = count_params(model,mode='all')
	model.config['trainable params'] = num_params
	model.config['all params'] = all_num_params
        with tf.Session(graph=model.graph) as sess:
	    ckpt = tf.train.get_checkpoint_state(args.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(sess, ckpt.model_checkpoint_path)
    	        print('Model restored from:'+args.save_dir) 
            batchErrors = np.zeros(len(batchedData))
            batchRandIxs = np.random.permutation(len(batchedData)) 
            for batch, batchOrigI in enumerate(batchRandIxs):
                batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                feedDict = {model.inputX: batchInputs,
			    model.targetIxs: batchTargetIxs,
			    model.targetVals: batchTargetVals,
			    model.targetShape: batchTargetShape,
			    model.seqLengths: batchSeqLengths}

                l, er, lmt, pre, y = sess.run([model.loss, model.errorRate,
					    model.logitsMaxTest,
					    model.predictions,
					    model.targetY],
				            feed_dict=feedDict)


		er = get_edit_distance([pre.values],[y.values],mode='test')
	    	print(output_to_sequence(pre,mode='phoneme'))
                print('Minibatch', batch+1, 'test error rate:', er)
            	batchErrors[batch] = er*len(batchSeqLengths)
            epochER = batchErrors.sum() / totalN
            print('TIMIT test error rate:', epochER)
	    logging(model,self.logfile,epochER,mode='test')

if __name__ == '__main__':
    tr = Trainer()
    if tr.args.mode == 'train':
	print('train mode')
	tr.train()
    elif tr.args.mode == 'test':
	print('test mode')
        tr.test()
    else:
	pass
