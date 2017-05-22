#-*- coding:utf-8 -*-

#!/usr/bin/python
''' Automatic Speech Recognition Evaluator

author(s):
nemik
     
date:2017-5-21
'''

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import time
import datetime
import os
from six.moves import cPickle
from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.contrib.rnn.python.ops.core_rnn import static_bidirectional_rnn

from utils.utils import load_batched_data
from utils.utils import data_lists_to_batches
from utils.utils import describe
from utils.utils import getAttrs
from utils.utils import output_to_sequence
from utils.utils import list_dirs
from utils.utils import logging
from utils.utils import count_params
from utils.utils import target2phoneme
from utils.utils import get_edit_distance
from utils.taskUtils import get_num_classes
from utils.taskUtils import check_path_exists
from utils.taskUtils import dotdict
from utils.functionDictUtils import model_functions_dict
from utils.functionDictUtils import activation_functions_dict
from utils.functionDictUtils import optimizer_functions_dict

from models.resnet import ResNet
from models.brnn import BiRNN
from models.dynamic_brnn import DBiRNN

from tensorflow.python.platform import flags
from tensorflow.python.platform import app

# for preprocess

from feature.core.sigprocess import *
from feature.core.calcmfcc import calcfeat_delta_delta
import scipy.io.wavfile as wav
import numpy as np
import sklearn
from sklearn import preprocessing

flags.DEFINE_string('wav_file', '', 'location of wav file to do prediction with')
flags.DEFINE_string('model_dir', '', 'path to saved checkpoint of trained model to use for prediction')

flags.DEFINE_string('preprocess_mode', 'mfcc', 'mfcc or fbank mode for preprocess')
flags.DEFINE_float('winlen', 0.02, 'specify the window length of feature')
flags.DEFINE_float('winstep', 0.01, 'specify the window step length of feature')
flags.DEFINE_integer('featlen', 13, 'Features length')

flags.DEFINE_string('level', 'cha', 'set the task level, phn, cha, or seq2seq, seq2seq will be supported soon')

flags.DEFINE_string('model', 'DBiRNN', 'set the model to use, DBiRNN, BiRNN, ResNet..')
flags.DEFINE_string('rnncell', 'lstm', 'set the rnncell to use, rnn, gru, lstm...')
flags.DEFINE_integer('num_layer', 2, 'set the layers for rnn')
flags.DEFINE_string('activation', 'tanh', 'set the activation to use, sigmoid, tanh, relu, elu...')
flags.DEFINE_string('optimizer', 'adam', 'set the optimizer to use, sgd, adam...')

#flags.DEFINE_integer('batch_size', 64, 'set the batch size')
flags.DEFINE_integer('batch_size', 1, 'set the batch size')
flags.DEFINE_integer('num_hidden', 256, 'set the hidden size of rnn cell')
#flags.DEFINE_integer('num_feature', 60, 'set the size of input feature')
flags.DEFINE_integer('num_feature', 39, 'set the size of input feature')
flags.DEFINE_integer('num_classes', 30, 'set the number of output classes')
flags.DEFINE_float('lr', 0.0001, 'set the learning rate')
flags.DEFINE_float('dropout_prob', 0.1, 'set probability of dropout')
flags.DEFINE_float('grad_clip', 1, 'set the threshold of gradient clipping, -1 denotes no clipping')

FLAGS = flags.FLAGS

level = FLAGS.level

model_dir = FLAGS.model_dir

preprocess_mode = FLAGS.preprocess_mode
winlen = FLAGS.winlen
winstep = FLAGS.winstep
featlen = FLAGS.featlen


model_fn = model_functions_dict[FLAGS.model]
rnncell = FLAGS.rnncell
num_layer = FLAGS.num_layer

activation_fn = activation_functions_dict[FLAGS.activation]
optimizer_fn = optimizer_functions_dict[FLAGS.optimizer]

batch_size = FLAGS.batch_size
num_hidden = FLAGS.num_hidden
num_feature = FLAGS.num_feature
num_classes = get_num_classes(level)
grad_clip = FLAGS.grad_clip
lr = FLAGS.lr

keep_prob = 1-FLAGS.dropout_prob

# first preprocess the input file
# 9 inputs, this is the one being used
# returns the features
def wav2feature(in_wav, win_len, win_step, mode, feature_len, seq2seq=False):
    if in_wav.endswith('.wav'):
        rate = None
        sig = None
        try:
            (rate,sig)= wav.read(in_wav)
        except ValueError as e:
            if e.message == "File format 'NIST'... not understood.":
                sf = Sndfile(in_wav, 'r')
                nframes = sf.nframes
                sig = sf.read_frames(nframes)
                rate = sf.samplerate
        feat = calcfeat_delta_delta(sig,rate,win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
        feat = preprocessing.scale(feat)
        feat = np.transpose(feat)
        return feat

wav_file = FLAGS.wav_file
feats = wav2feature(wav_file, winlen, winstep, preprocess_mode, featlen)

def _default_configs():
    return {'level': level,
            'rnncell': rnncell,
            'batch_size': batch_size,
            'num_hidden': num_hidden,
            'num_feature': num_feature,
            'num_class': num_classes,
            'num_layer': num_layer,
            'activation': activation_fn,
            'optimizer': optimizer_fn,
            'learning_rate': lr,
            'keep_prob': keep_prob,
            'grad_clip': grad_clip,
           }
args_dict = _default_configs()
args = dotdict(args_dict)

batchedData, maxTimeSteps = data_lists_to_batches([np.array(feats)], [[np.array(0)]], batch_size, level)

model = model_fn(args, maxTimeSteps)
num_params = count_params(model, mode='trainable')
all_num_params = count_params(model, mode='all')
model.config['trainable params'] = num_params
model.config['all params'] = all_num_params
print(model.config)

with tf.Session(graph=model.graph) as sess:
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored from:' + model_dir)
    else:
        print("please select a path for an existing model")
        sys.exit(1)

    batchErrors = np.zeros(len(batchedData))
    batchRandIxs = np.random.permutation(len(batchedData))

    for batch, batchOrigI in enumerate(batchRandIxs):
        batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
        batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
        feedDict = {model.inputX: batchInputs, model.targetIxs: batchTargetIxs,
                    model.targetVals: batchTargetVals, model.targetShape: batchTargetShape,
                    model.seqLengths: batchSeqLengths}

        #l, pre, y, er = sess.run([model.loss, model.predictions, 
        #    model.targetY, model.errorRate], feed_dict=feedDict)
        pre = sess.run([model.predictions], feed_dict=feedDict)
        result = output_to_sequence(pre[0], type=level)
        print("\nRESULT: {}\n".format(result))
