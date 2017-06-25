#-*- coding:utf-8 -*-
#!/usr/bin/python

''' This library provides some common functions
author:
zzw922cn
liujq

date:2016-11-09
fix: 2017-5-13
'''
import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import time
from functools import wraps
import os
from glob import glob
import numpy as np
import tensorflow as tf
import math


def describe(func):
    ''' wrap function,to add some descriptions for function and its running time
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__+'...')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(str(func.__name__+' in '+ str(end-start)+' s'))
        return result
    return wrapper

def getAttrs(object, name):
    ''' get attributes for object
    '''
    assert type(name) == list, 'name must be a list'
    value = []
    for n in name:
        value.append(getattr(object, n, 'None'))
    return value

def setAttrs(object, attrsName, attrsValue):
    ''' register attributes for this class '''
    assert type(attrsName) == list, 'attrsName must be a list'
    assert type(attrsValue) == list, 'attrsValue must be a list'
    for name, value in zip(attrsName, attrsValue):
        object.__dict__[name] = value

def output_to_sequence(lmt, type='phn'):
    ''' convert the output into sequences of characters or phonemes
    '''
    phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h',
       'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl',
       'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',
       'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#',
       'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',
       'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',
       'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
       'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',
       'v', 'w', 'y', 'z', 'zh']
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
    if type == 'phn':
        seq = []
        for ind in indexes:
            if ind == len(phn):
                pass
            else:
                seq.append(phn[ind])
        seq = ' '.join(seq)
        return seq

    elif type == 'cha':
        seq = []
        for ind in indexes:
            if ind == 0:
                seq.append(' ')
            elif ind == 27:
                seq.append("'")
            elif ind == 28:
                pass
            else:
                seq.append(chr(ind+96))
        seq = ''.join(seq)
        return seq
    else:
        raise TypeError('mode should be phoneme or character')

def target2phoneme(target):
    seq = []
    for t in target:
        if t == len(phn):
            pass
        else:
            seq.append(phn[t])
    seq = ' '.join(seq)
    return seq

@describe
def logging(model,logfile,errorRate,epoch=0,delta_time=0,mode='train'):
    ''' log the cost and error rate and time while training or testing
    '''
    if mode != 'train' and mode != 'test' and mode != 'config' and mode != 'dev':
        raise TypeError('mode should be train or test or config.')
    logfile = logfile
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
    elif mode == 'dev':
        logfile = logfile+'_DEV'
        with open(logfile, "a") as myfile:
            myfile.write(str(model.config)+'\n')
            myfile.write(str(time.strftime('%X %x %Z'))+'\n')
            myfile.write("development error rate:"+str(errorRate)+'\n')

@describe
def count_params(model, mode='trainable'):
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

def list_to_sparse_tensor(targetList, level):
    ''' turn 2-D List to SparseTensor
    '''
    indices = [] #index
    vals = [] #value
    assert level == 'phn' or level == 'cha', 'type must be phoneme or character, seq2seq will be supported in future'
    phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h',\
       'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl',\
       'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',\
       'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#',\
       'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',\
       'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',\
       'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',\
       'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',\
       'v', 'w', 'y', 'z', 'zh']

    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',\
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',\
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',\
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',\
                 'v', 'w', 'y', 'z', 'zh']


    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',\
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',\
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',\
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',\
                 'v', 'w', 'y', 'z', 'zh']

    if level == 'cha':
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(axis=0)[1]+1] #shape
        return (np.array(indices), np.array(vals), np.array(shape))

    elif level == 'phn':
        '''
        for phn level, we should collapse 61 labels into 39 labels before scoring
        
        Reference:
          Heterogeneous Acoustic Measurements and Multiple Classifiers for Speech Recognition(1986), 
            Andrew K. Halberstadt, https://groups.csail.mit.edu/sls/publications/1998/phdthesis-drew.pdf
        '''
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                if val < len(phn) and (phn[val] in mapping.keys()):
                    val = group_phn.index(mapping[phn[val]])
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(0)[1]+1] #shape
        return (np.array(indices), np.array(vals), np.array(shape))

    else:
        ##support seq2seq in future here
        raise ValueError('Invalid level: %s'%str(level))

def get_edit_distance(hyp_arr, truth_arr, normalize, level):
    ''' calculate edit distance
    This is very universal, both for cha-level and phn-level
    '''

    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.reduce_sum(tf.edit_distance(hyp, truth, normalize=normalize))

    with tf.Session(graph=graph) as session:
        truthTest = list_to_sparse_tensor(truth_arr, level)
        hypTest = list_to_sparse_tensor(hyp_arr, level)
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run(editDist, feed_dict=feedDict)
    return dist

def data_lists_to_batches(inputList, targetList, batchSize, level, maxlen=0):
    ''' padding the input list to a same dimension, integrate all data into batchInputs
    '''
    assert len(inputList) == len(targetList)
    # dimensions of inputList:batch*39*time_length

    nFeatures = inputList[0].shape[0]
    maxLength = maxlen
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
        dataBatches.append((batchInputs, list_to_sparse_tensor(batchTargetList, level), batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxLength)

def load_batched_data(mfccPath, labelPath, batchSize, mode, level):
    '''returns 3-element tuple: batched data (list), maxTimeLength (int), and
       total number of samples (int)'''
    return data_lists_to_batches([np.load(os.path.join(mfccPath, fn)) for fn in os.listdir(mfccPath)],
                                 [np.load(os.path.join(labelPath, fn)) for fn in os.listdir(labelPath)],
                                 batchSize, level) + \
            (len(os.listdir(mfccPath)),)

def list_dirs(mfcc_dir, label_dir):
    mfcc_dirs = glob(mfcc_dir)
    label_dirs = glob(label_dir)
    for mfcc,label in zip(mfcc_dirs,label_dirs):
        yield (mfcc,label)


def build_weight(shape, name, func='he_normal', range=None):
    """ Initializes weight.
    supply following initialization:
	xavier initialization
	random normalization
	uniform
    We also add the L2 loss to weight for training.
    """
    if type(shape) is not list:
        raise TypeError('shape must be a list')

    initializer = tf.constant_initializer()
    if func == 'xavier':
        fan_in, fan_out = _get_dims(shape)
        range = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-range, range)

    elif func == 'he_normal':
        fan_in, _ = _get_dims(shape)
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev=std)

    elif func == 'normal':
        initializer = tf.random_normal_initializer(stddev=0.1)

    elif func == 'uniform':
        if range is None:
            raise ValueError("range must be specified if you use uniform initialization")
        initializer = tf.random_uniform_initializer(-range, range)

    with tf.variable_scope(name or "weight_var"):
        var = tf.get_variable(name, shape, initializer=initializer)
    tf.add_to_collection('l2', tf.nn.l2_loss(var))
    return var


def build_forward_layer(inpt, shape, kernel='relu', name_scope='fc1'):
    fc_w = build_weight(shape,name=name_scope+'_w')
    fc_b = build_weight([shape[1]],name=name_scope+'_b')
    if kernel == 'relu':
        fc_h = tf.nn.relu(tf.matmul(inpt,fc_w) + fc_b)
    elif kernel == 'elu':
        fc_h = tf.nn.elu(tf.matmul(inpt,fc_w) + fc_b)
    elif kernel == 'linear':
        fc_h = tf.matmul(inpt,fc_w) + fc_b
    return fc_h

def build_conv_layer(inpt, filter_shape, stride,name=None):
    ''' build a convolutional layer with batch normalization
    '''
    # BN->ReLU->conv
    # 1.batch normalization
    in_channels = filter_shape[2]
    mean, var = tf.nn.moments(inpt, axes=[0,1,2])
    with tf.variable_scope(name or "conv_layer"):
        beta = tf.Variable(tf.zeros([in_channels]), name="beta")
        gamma = build_weight([in_channels], name="gamma")
    batch_normed = tf.nn.batch_normalization(
    				inpt,
				    mean, var,
				    beta, gamma,
				    0.001,
				    name=name+'_bn')
    # 2.relu
    activated = tf.nn.relu(batch_normed)
    # 3.convolution
    filter_ = build_weight(filter_shape, name=name+'_filter')
    output = tf.nn.conv2d(activated, filter=filter_, strides=[1, stride, stride, 1], padding='SAME')
    output = tf.nn.dropout(output, keep_prob=0.6)
    return output

def batch_norm(x, is_training=True):
    """ Batch normalization.
    """
    with tf.variable_scope('BatchNorm'):
        inputs_shape = x.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        param_shape = inputs_shape[-1:]

        beta = tf.get_variable('beta', param_shape, initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable('gamma', param_shape, initializer=tf.constant_initializer(1.))
        batch_mean, batch_var = tf.nn.moments(x, axis)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def _get_dims(shape):
    """get shape for initialization
    """
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    return fan_in, fan_out

def dropout(x, keep_prob, is_training):
    """ Apply dropout to a tensor
    """
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob, is_training=is_training)


