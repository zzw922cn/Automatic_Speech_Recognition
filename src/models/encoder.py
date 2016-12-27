# -*- coding:utf-8 -*-
''' encoder for local-global memory network

encoder can be a RNN cell and its output will be taken
as input for rnn_decoder.
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
     
date:2016-12-07
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  
from six.moves import zip    

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

linear = rnn_cell._linear  # pylint: disable=protected-access


def rnn_encoder(encoder_inputs, initial_state, cell, scope=None):
    ''' Recurrent neural network for encoder

    The cell can be rnn, gru or lstm.
    We can encode the network into a feature vector representation
    args:
        encoder_inputs: 2-D List of shape [batch_size,feature_size]
        initial_state: initial hidden state for the rnn cell
        cell: type of rnn, eg: rnn, gru, lstm, or any other you build
        scope: VariableScope for the created subgraph; defaults to "rnn_encoder"
    '''
    with tf.variable_scope(scope or "rnn_encoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(encoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
    return outputs, state
            
def cnn_encoder(encoder_inputs, initial_state, cell, scope=None):
    
    
    
    
