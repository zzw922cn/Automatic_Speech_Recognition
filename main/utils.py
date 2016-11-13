#-*- coding:utf-8 -*-
#!/usr/bin/python

''' This file is designed to provide some common tools
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

import numpy as np
import tensorflow as tf
import time
from functools import wraps
import os
from glob import glob

def describe(func):
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

def output_to_sequence(lmt):
    output = np.unique(lmt) 
    sequence = []
    for o in output:
        if o==0:
            sequence.append(' ')
        elif o==27:
	    pass
        else:
            sequence.append(chr(o+96))
    sequence = ''.join(sequence)
    return sequence

def list_to_sparse_tensor(targetList):
    ''' 将二维List变成SparseTensor，为了适应Edit distance API
    '''

    indices = [] #索引
    vals = [] #变量 
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1] #形状
    return (np.array(indices), np.array(vals), np.array(shape))

def get_edit_distance(hyp_arr,truth_arr):
    ''' 得到两个列表的编辑距离
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
    # dimensions of inputList:batch*39*time_length 
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

        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]

        batchInputs = np.zeros((maxLength, batchSize, nFeatures))

        batchTargetList = []

        for batchI, origI in enumerate(randIxs[start:end]):
	    # padSecs is the length of padding  
            padSecs = maxLength - inputList[origI].shape[1]
	    # numpy.pad pad the inputList[origI] with zeos at the tail
            batchInputs[:,batchI,:] = np.pad(inputList[origI].T, ((0,padSecs),(0,0)),
                                             'constant', constant_values=0)
	    # target label
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, list_to_sparse_tensor(batchTargetList),
                          batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxLength)

def load_batched_data(specPath, targetPath, batchSize):
    import os
    '''returns 3-element tuple: batched data (list), max # of time steps (int), and
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

if __name__=='__main__':
    for (mfcc,label) in list_dirs('/home/pony/github/data/label/*/','/home/pony/github/data/mfcc/*/'):
	print mfcc
	print label
