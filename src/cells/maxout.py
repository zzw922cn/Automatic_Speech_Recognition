#!/usr/bin/env python
# -*- coding:utf-8 -*-

''' maxout network for lyrics generation or automatic speech recognition
author: 
Jiaqi Liu &

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
     
date:2016-12-16
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def maxout_cnn_layer(inputs, filter, bind_num,strides,
                    padding, name=None):
    ''' maxout convolutional layer
    maxout_cnn_layer(inputs, filter, bind_num, strides, padding)

        inputs: tensor with shape:[batch size,...,channel]
        filter: a filter variable with shape:[width,height,in_channel,out_channel]
        bind_num: indicate how many feature maps we want to bind in a group
        strides: strides - see conv2d
        padding: padding - see conv2d

    return: maxout cnn layer calculated tensor
    reference: 'Maxout Networks' -Ian Goodfellow etc.
    '''

    conv = tf.nn.conv2d(input=inputs,
                filter=filter,
                strides=strides,
                padding=padding,
		name=name)
    return maxout(conv, bind_num)

def maxout_weights(filter_shape,bind_num):
    ''' Weight intialization for maxout cnn layer. Automatically expand filter
    size for binding.

    filter_shape: the real filter shape we like to have.
    '''
    filter_shape[-1]=filter_shape[-1]*bind_num
    return tf.Variable(tf.random_normal(filter_shape, stddev=0.01))

def maxout(inputs, bind_num):
    '''
    Raw cnn maxout activation function.
        Maxout activation function for cnn returns maximun value of binded feature maps.
        For examples: if we want to get 16 feature maps, we can set filter shape as
        [width,height,in_channel,32] with bind_num=2. It divided 32 feature maps into 16
        groups, and get the maximun value of each weight in each groups.

    maxout(inputs, bind_num)
        inputs: input tensor with shape [batch_size, ..., in_chan]
        bind_num: indicate how many feature maps we want to bind in a group

        return: tensor with shape:[batch_size, ..., in_chan//bind_num].
    '''
    shape = inputs.get_shape().as_list()
    in_chan = shape[-1]
    shape[-1] = in_chan//bind_num
    shape += [bind_num]
    shape[0]=-1
    return tf.reduce_max(tf.reshape(inputs,shape),-1,keep_dims=False)

