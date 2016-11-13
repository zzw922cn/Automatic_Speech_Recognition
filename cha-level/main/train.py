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
from model import Model
from utils import getAttrs
from utils import output_to_sequence
from utils import list_dirs

class Trainer(object):
    
    def __init__(self):
	parser = argparse.ArgumentParser()
        parser.add_argument('--mfcc_dir', type=str, default='/home/pony/github/data/mfcc/0/',
                       help='data directory containing mfcc numpy files, usually end with .npy')

        parser.add_argument('--label_dir', type=str, default='/home/pony/github/data/label/0/',
                       help='data directory containing label numpy files, usually end with .npy')

        parser.add_argument('--model', type=str, default='blstm',
                       help='set the model of ASR, ie: blstm')

	parser.add_argument('--keep', type=bool, default=False,
                       help='train the model based on model saved')


	parser.add_argument('--save', type=bool, default=True,
                       help='to save the model in the disk')

	parser.add_argument('--evaluation', type=bool, default=False,
                       help='train the model based on model saved')

        parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='set the step size of each iteration')

        parser.add_argument('--num_epoch', type=int, default=200000,
                       help='set the total number of training epochs')

        parser.add_argument('--batch_size', type=int, default=1,
                       help='set the number of training samples in a mini-batch')

        parser.add_argument('--num_feature', type=int, default=39,
                       help='set the dimension of feature, ie: 39 mfccs, you can set 39 ')

        parser.add_argument('--num_hidden', type=int, default=256,
                       help='set the number of neurons in hidden layer')

        parser.add_argument('--num_class', type=int, default=28,
                       help='set the number of labels in the output layer')

        parser.add_argument('--save_dir', type=str, default='/home/pony/github/ASR_libri/libri/cha-level/save/',
                       help='set the directory to save the model, containing checkpoint file and parameter file')

        parser.add_argument('--restore_from', type=str, default='/home/pony/github/ASR_libri/libri/cha-level/save/',
                       help='set the directory of check_point path')

        parser.add_argument('--model_checkpoint_path', type=str, default='/home/pony/github/ASR_libri/libri/cha-level/save/model.ckpt-0',
                       help='set the directory to restore the model, containing checkpoint file and parameter file')

	self.args = parser.parse_args()
	self.start(self.args)

    @describe
    def load_data(self,args):
        return load_batched_data(args.mfcc_dir,args.label_dir,args.batch_size)

    def start(self,args):
	# load data
        batchedData, maxTimeSteps, totalN = self.load_data(args)

	# build graph
	model = Model(args,maxTimeSteps)
        with tf.Session(graph=model.graph) as sess:
    	    print('Initializing')
    	    sess.run(model.initial_op)
	    if args.keep == True:
		ckpt = tf.train.get_checkpoint_state(args.restore_from)
		model_checkpoint_path = args.model_checkpoint_path
                if ckpt and ckpt.model_checkpoint_path:
                    model.saver.restore(model.sess, ckpt.model_checkpoint_path)
    		    print('Model restored from:'+args.restore_from) 
    
    	    for epoch in range(args.num_epoch):
		start = time.time()
        	print('Epoch', epoch+1, '...')
        	batchErrors = np.zeros(len(batchedData))
        	batchRandIxs = np.random.permutation(len(batchedData)) 
        	for batch, batchOrigI in enumerate(batchRandIxs):
            	    batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                    batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                    feedDict = {model.inputX: batchInputs, model.targetIxs: batchTargetIxs, model.targetVals: batchTargetVals,
                            model.targetShape: batchTargetShape, model.seqLengths: batchSeqLengths}
		    if args.evaluation == True:
                        l, er, lmt = sess.run([model.loss, model.errorRate, model.logitsMaxTest], feed_dict=feedDict)
		    else:
                        _, l, er, lmt = sess.run([model.optimizer, model.loss, model.errorRate, model.logitsMaxTest], feed_dict=feedDict)
                   
	    	    print(output_to_sequence(lmt))
                    if (batch % 1) == 0:
                	print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                	print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            	    batchErrors[batch] = er*len(batchSeqLengths)
		    
		    if (args.save==True) and  ((epoch*len(batchRandIxs)+batch+1)%1000==0 or (epoch==args.num_epoch-1 and batch==len(batchRandIxs)-1)):
		        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        save_path = model.saver.save(sess,checkpoint_path,global_step=epoch)
                        print('Model has been saved in:'+str(save_path))
	        end = time.time()
		print('Epoch '+str(epoch+1)+' needs time:'+str(end-start)+' s')	
		if args.save==True and (epoch+1)%1==0:
		    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    save_path = model.saver.save(sess,checkpoint_path,global_step=epoch)
                    print('Model has been saved in file')
                epochErrorRate = batchErrors.sum() / totalN
                print('Epoch', epoch+1, 'error rate:', epochErrorRate)

if __name__ == '__main__':
    t=Trainer()
