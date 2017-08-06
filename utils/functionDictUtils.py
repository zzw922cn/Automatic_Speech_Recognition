#-*- coding:utf-8 -*-
#!/usr/bin/python
''' Function dict for convenience

author(s):
zzw922cn
     
date:2017-4-15
'''

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import tensorflow as tf 

from models.dynamic_brnn import DBiRNN
from models.deepSpeech2 import DeepSpeech2

model_functions_dict = {'DBiRNN': DBiRNN, 'deepSpeech2': DeepSpeech2}

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

