#-*- coding:utf-8 -*-
#!/usr/bin/python

''' This library provides some common functions
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

common function list:

	describe(func)
	getAttrs(object,name)
	setAttrs(object,attrsName,attrsValue)
	output_to_sequence(lmt,mode='phoneme')
	target2phoneme(target)
	logging(model,errorRate,epoch=0,delta_time=0,mode='train')
	count_params(model,mode='trainable')
	list_to_sparse_tensor(targetList)
	get_edit_distance(hyp_arr,truth_arr)
	data_lists_to_batches(inputList, targetList, batchSize)
	load_batched_data(specPath, targetPath, batchSize)
	list_dirs(mfcc_dir,label_dir)
	build_weight(shape,name=None,func='truncated_normal')
	build_forward_layer(inpt,shape,kernel='relu',name_scope='fc1')
	build_conv_layer(inpt,filter_shape,stride,name=None)	
'''
import time
from functools import wraps
import os
from glob import glob
import numpy as np
import tensorflow as tf



def describe(func):
    ''' wrap function,to add some descriptions for function and its running time
    '''
    @wraps(func)
    def wrapper(*args,**kwargs):
        print(func.__name__+'...')
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        print(str(func.__name__+' in '+ str(end-start)+' s'))
        return result
    return wrapper

def getAttrs(object,name):
    ''' get attributes for object
    '''
    assert type(name)==list, 'name must be a list'
    value = []
    for n in name:
        value.append(getattr(object,n,'None'))
    return value

def setAttrs(object,attrsName,attrsValue):
    ''' register attributes for this class '''
    assert type(attrsName)==list, 'attrsName must be a list'
    assert type(attrsValue)==list, 'attrsValue must be a list'
    for name,value in zip(attrsName,attrsValue):
        object.__dict__[name]=value

def output_to_sequence(lmt,mode='phoneme'):
    ''' convert the output into sequences of characters or phonemes
    '''
    sequences = []
    start = 0
    sequences.append([])
    for i in range(len(lmt[0])):
	if lmt[0][i][0] == start:
	    sequences[start].append(lmt[1][i])
	else:
	    start = start + 1
            sequences.append([])

    #here, we only print the first sequence of batch
    indexes = sequences[0] #here, we only print the first sequence of batch
    if mode=='phoneme':
	phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
	seq = []
        for ind in indexes:
            if ind==len(phn):
                pass
            else:
                seq.append(phn[ind])
        seq = ' '.join(seq)
        return seq
	
    elif mode=='character':
        seq = []
        for ind in indexes:
            if ind==0:
                seq.append(' ')
            elif ind==27:
	        pass
            else:
                seq.append(chr(ind+96))
        seq = ''.join(seq)
        return seq
    else:
	raise TypeError('mode should be phoneme or character')

def target2phoneme(target):
    phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
    seq = []
    for t in target:
	if t==len(phn):
	    pass
	else:
	    seq.append(phn[t])
    seq = ' '.join(seq)
    return seq

@describe
def logging(model,errorRate,epoch=0,delta_time=0,mode='train'):
    ''' log the cost and error rate and time while training or testing
    '''
    if mode != 'train' and mode!='test' and mode!='config':
	raise TypeError('mode should be train or test or config.')
    logfile = model.logfile
    if mode == 'config':
	with open(logfile, "a") as myfile:
            myfile.write(str(model.config)+'\n')
    elif mode == 'train':
        with open(logfile, "a") as myfile:
	    myfile.write(str(time.strftime('%X %x %Z'))+'\n')
    	    myfile.write("Epoch:"+str(epoch+1)+' '+"train error rate:"+str(errorRate)+'\n')
    	    myfile.write("Epoch:"+str(epoch+1)+' '+"train time:"+str(delta_time)+' s\n')
    elif mode == 'test':
        logfile = logfile+'_TEST'
        with open(logfile, "a") as myfile:
            myfile.write(str(model.config)+'\n')
	    myfile.write(str(time.strftime('%X %x %Z'))+'\n')
    	    myfile.write("test error rate:"+str(errorRate)+'\n')

@describe
def count_params(model,mode='trainable'):
    ''' count all parameters of a tensorflow graph
    '''
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
	raise TypeError('mode should be all or trainable.')
    print('number of '+mode+' parameters: '+str(num))
    return num

def list_to_sparse_tensor(targetList):
    ''' turn 2-D List to SparseTensor
    '''

    indices = [] #index
    vals = [] #value 
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1] #shape
    return (np.array(indices), np.array(vals), np.array(shape))

def get_edit_distance(hyp_arr,truth_arr):
    ''' calculate edit distance 
    '''
    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.edit_distance(hyp, truth, normalize=True)

    with tf.Session(graph=graph) as session:
        truthTest = list_to_sparse_tensor(truth_arr)
        hypTest = list_to_sparse_tensor(hyp_arr)
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run(editDist, feed_dict=feedDict)
    return dist

def data_lists_to_batches(inputList, targetList, batchSize):
    ''' padding the input list to a same dimension, integrate all data into batchInputs
    '''
    assert len(inputList) == len(targetList)
    # dimensions of inputList:batch*39*time-length
    nFeatures = inputList[0].shape[0]
    maxLength = 0
    for inp in inputList:
	# find the max time_length
        maxLength = max(maxLength, inp.shape[1])
    # randIxs is the shuffled index from range(0,len(inputList)) 
    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
	# batchSeqLengths store the time-length of each sample in a mini-batch
        batchSeqLengths = np.zeros(batchSize)
  	# randIxs is the shuffled index of input list
	for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]

        batchInputs = np.zeros((maxLength, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
	    # padSecs is the length of padding  
            padSecs = maxLength - inputList[origI].shape[1]
	    # numpy.pad pad the inputList[origI] with zeos at the tail
            batchInputs[:,batchI,:] = np.pad(inputList[origI].T, ((0,padSecs),(0,0)), 'constant', constant_values=0)
	    # target label
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, list_to_sparse_tensor(batchTargetList), batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxLength)

def load_batched_data(specPath, targetPath, batchSize):
    import os
    '''returns 3-element tuple: batched data (list), maxTimeLength (int), and
       total number of samples (int)'''
    return data_lists_to_batches([np.load(os.path.join(specPath, fn)) for fn in os.listdir(specPath)],
                                 [np.load(os.path.join(targetPath, fn)) for fn in os.listdir(targetPath)],
                                 batchSize) + \
            (len(os.listdir(specPath)),)

def list_dirs(mfcc_dir,label_dir):
    mfcc_dirs = glob(mfcc_dir)
    label_dirs = glob(label_dir)
    for mfcc,label in zip(mfcc_dirs,label_dirs):
	yield (mfcc,label)

def build_weight(shape,name=None,func='truncated_normal'):
    if type(shape) is not list:
	raise TypeError('shape must be a list')
    if func == 'truncated_normal':
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32,name = name))

def build_forward_layer(inpt,shape,kernel='relu',name_scope='fc1'):
    fc_w = build_weight(shape,name=name_scope+'_w')
    fc_b = build_weight([shape[1]],name=name_scope+'_b')
    if kernel == 'relu':
	fc_h = tf.nn.relu(tf.matmul(inpt,fc_w) + fc_b)
    elif kernel == 'elu':
	fc_h = tf.nn.elu(tf.matmul(inpt,fc_w) + fc_b)
    elif kernel == 'linear':
	fc_h = tf.matmul(inpt,fc_w) + fc_b
    return fc_h

def build_conv_layer(inpt,filter_shape,stride,name=None):
    # BN->ReLU->conv
    # 1.batch normalization
    in_channels = filter_shape[2]
    mean, var = tf.nn.moments(inpt, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([in_channels]), name="beta")
    gamma = build_weight([in_channels], name="gamma")
    batch_norm = tf.nn.batch_normalization(
    				inpt, 
				mean, var, 
				beta, gamma, 
				0.001,
				name=name+'_bn')
    # 2.relu
    activated = tf.nn.relu(batch_norm)
    # 3.convolution
    filter_ = build_weight(filter_shape,name=name+'_filter')
    output = tf.nn.conv2d(activated, filter=filter_, strides=[1, stride, stride, 1], padding='SAME')
    output = tf.nn.dropout(output,keep_prob=0.6)
    return output

# test code
if __name__=='__main__':
    for (mfcc,label) in list_dirs('/home/pony/github/data/label/*/','/home/pony/github/data/mfcc/*/'):
	print mfcc
	print label
