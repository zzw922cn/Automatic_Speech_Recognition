# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : functionDictUtils.py
# Description  : Common function dictionaries for Automatic Speech Recognition
# ******************************************************


import tensorflow as tf 

activation_functions_dict = {
    'sigmoid': tf.sigmoid, 'tanh': tf.tanh, 'relu': tf.nn.relu, 'relu6': tf.nn.relu6,
    'elu': tf.nn.elu, 'softplus': tf.nn.softplus, 'softsign': tf.nn.softsign
    # for detailed intro, go to https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_
    }

optimizer_functions_dict = {
    'gd': tf.train.GradientDescentOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adam': tf.train.AdamOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer
    }

