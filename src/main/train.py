#-*- coding:utf-8 -*-
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

import argparse
import time
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
from utils import getAttrs
from utils import output_to_sequence
from utils import list_dirs
from utils import logging
from utils import count_params
from utils import target2phoneme

from resnet import ResNet
from brnn import BiRNN

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

	parser.add_argument('--log_dir', type=str, default='/home/pony/github/Automatic-Speech-Recognition/log/timit/',
                       help='directory to log events while training')

	parser.add_argument('--model', default='BiRNN',
		       help='model for ASR')

	parser.add_argument('--rnncell', default='rnn',
		       help='rnn cell, 3 choices:rnn,lstm,gru')

        parser.add_argument('--num_layer', type=int, default=2,
                       help='set the number of hidden layer or bidirectional layer')

        parser.add_argument('--activation', default=tf.nn.elu,
                       help='set the activation function of each layer')

        parser.add_argument('--optimizer', type=type, default=tf.train.AdamOptimizer,
                       help='set the optimizer to train the model,eg:AdamOptimizer,GradientDescentOptimizer')
	
        parser.add_argument('--grad_clip', default=-1,
                       help='set gradient clipping when backpropagating errors')

	parser.add_argument('--keep', type=bool, default=False,
                       help='train the model based on model saved')

	parser.add_argument('--save', type=bool, default=True,
                       help='to save the model in the disk')

	parser.add_argument('--evaluation', type=bool, default=False,
                       help='test the model based on trained parameters, but at present, we can"t test during training.')

        parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='set the step size of each iteration')

        parser.add_argument('--num_epoch', type=int, default=200000,
                       help='set the total number of training epochs')

        parser.add_argument('--batch_size', type=int, default=64,
                       help='set the number of training samples in a mini-batch')

        parser.add_argument('--test_batch_size', type=int, default=1,
                       help='set the number of testing samples in a mini-batch')

        parser.add_argument('--num_feature', type=int, default=39,
                       help='set the dimension of feature, ie: 39 mfccs, you can set 39 ')

        parser.add_argument('--num_hidden', type=int, default=128,
                       help='set the number of neurons in hidden layer')

        parser.add_argument('--num_class', type=int, default=62,
                       help='set the number of labels in the output layer, if timit phonemes, it is 62; if timit characters, it is 28; if libri characters, it is 28')

        parser.add_argument('--save_dir', type=str, default='/home/pony/github/Automatic-Speech-Recognition/save/timit/',
                       help='set the directory to save the model, containing checkpoint file and parameter file')

        parser.add_argument('--restore_from', type=str, default='/home/pony/github/Automatic-Speech-Recognition/save/timit/',
                       help='set the directory of check_point path')

        parser.add_argument('--model_checkpoint_path', type=str, default='/home/pony/github/Automatic-Speech-Recognition/save/timit/model.ckpt-55',
                       help='set the directory to restore the model, containing checkpoint file and parameter file')

	self.args = parser.parse_args()

    @describe
    def load_data(self,args,mode='train'):
  	if mode == 'train':	
            return load_batched_data(args.train_mfcc_dir,args.train_label_dir,args.batch_size)
  	elif mode == 'test':	
            return load_batched_data(args.test_mfcc_dir,args.test_label_dir,args.test_batch_size)
	else:
	    raise TypeError('mode should be train or test.')

    def train(self):
	# load data
	args = self.args
        batchedData, maxTimeSteps, totalN = self.load_data(args,mode='train')

	if args.model == 'ResNet':
	    model = ResNet(args,maxTimeSteps)
	if args.model == 'BiRNN':
	    model = BiRNN(args,maxTimeSteps)
	num_params = count_params(model,mode='trainable')
	all_num_params = count_params(model,mode='all')
	model.config['trainable params'] = num_params
	model.config['all params'] = all_num_params
	print(model.config)
        with tf.Session(graph=model.graph) as sess:
	    if args.keep == True:
		ckpt = tf.train.get_checkpoint_state(args.restore_from)
		model_checkpoint_path = args.model_checkpoint_path
                if ckpt and ckpt.model_checkpoint_path:
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
    		    print('Model restored from:'+args.restore_from) 
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

                    _, l, er, lmt, pre = sess.run([model.optimizer, model.loss, model.errorRate, model.logitsMaxTest, model.predictions], feed_dict=feedDict)
	    	    print(output_to_sequence(pre,mode='phoneme'))

                    if (batch % 1) == 0:
                	print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                	print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            	    batchErrors[batch] = er*len(batchSeqLengths)
		    
		    if (args.save==True) and  ((epoch*len(batchRandIxs)+batch+1)%5000==0 or (epoch==args.num_epoch-1 and batch==len(batchRandIxs)-1)):
		        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        save_path = model.saver.save(sess,checkpoint_path,global_step=epoch)
                        print('Model has been saved in:'+str(save_path))
	        end = time.time()
		delta_time = end-start
		print('Epoch '+str(epoch+1)+' needs time:'+str(delta_time)+' s')	

		if args.save==True and (epoch+1)%1==0:
		    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    save_path = model.saver.save(sess,checkpoint_path,global_step=epoch)
                    print('Model has been saved in file')
                epochER = batchErrors.sum() / totalN
                print('Epoch', epoch+1, 'train error rate:', epochER)
	        logging(model,epochER,epoch,delta_time,mode='config')
		logging(model,epochER,epoch,delta_time,mode='train')
	
    def test(self):
	# load data
	args = self.args
        batchedData, maxTimeSteps, totalN = self.load_data(args,mode='test')
	model = resModel(args,maxTimeSteps)
	num_params = count_params(model,mode='trainable')
	all_num_params = count_params(model,mode='all')
	model.config['trainable params'] = num_params
	model.config['all params'] = all_num_params
        with tf.Session(graph=model.graph) as sess:
	    ckpt = tf.train.get_checkpoint_state(args.restore_from)
	    model_checkpoint_path = args.model_checkpoint_path
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(sess, ckpt.model_checkpoint_path)
    	        print('Model restored from:'+args.restore_from) 
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

                l, er, lmt, pre = sess.run([model.loss, model.errorRate,
					    model.logitsMaxTest,
					    model.predictions],
				            feed_dict=feedDict)
	    	print(output_to_sequence(pre,mode='phoneme'))
                print('Minibatch', batch, 'test loss:', l)
                print('Minibatch', batch, 'test error rate:', er)
            	batchErrors[batch] = er*len(batchSeqLengths)
            epochER = batchErrors.sum() / totalN
            print('TIMIT test error rate:', epochER)
	    logging(model,epochER,mode='test')

if __name__ == '__main__':
    tr = Trainer()
    if tr.args.evaluation is True:
	print('test mode')
	tr.test()
    else:
	print('train mode')
        tr.train()
