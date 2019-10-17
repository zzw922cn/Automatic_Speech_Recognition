#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : capsuleNetwork.py
# Author            : zewangzhang <zzw922cn@gmail.com>
# Date              : 17.10.2019
# Last Modified Date: 17.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : CapsuleNetwork.py
# Description  : Capsule network for Automatic Speech Recognition
# ******************************************************

import sys
import tensorflow as tf
import os
import numpy as np


def squashing(s):
    """ Squashing function for normalization
    size of s: [batch_size, 1, next_channels*next_num_capsules, next_output_vector_len, 1]
    size of v: [batch_size, 1, next_channels*next_num_capsules, next_output_vector_len, 1]
    """
    assert s.dtype == tf.float32
    squared_s = tf.reduce_sum(tf.square(s), axis=2, keep_dims=True)
    normed_s = tf.norm(s, axis=2, keep_dims=True)
    v = squared_s / (1+squared_s) / normed_s * s
    assert v.get_shape()==s.get_shape()
    return v

def routing(u, next_num_channels, next_num_capsules, next_output_vector_len, num_iter, scope=None):
    """ Routing algorithm for capsules of two adjacent layers
    size of u: [batch_size, channels, num_capsules, output_vector_len]
    size of w: [batch_size, channels, num_capsules, next_channels, next_num_capsules, vec_len, next_vec_len]
    """
    scope = scope or "routing"
    shape = u.get_shape()
    u = tf.reshape(u, [shape[0], shape[1], shape[2], 1, 1, shape[3], 1])
    u_ij = tf.tile(u, [1, 1, 1, next_num_channels, next_num_capsules, 1, 1])
    with tf.variable_scope(scope):
        w_shape = [1, shape[1], shape[2], next_num_channels, next_num_capsules, shape[3], next_output_vector_len]
        w = tf.get_variable("w", shape=w_shape, dtype=tf.float32)
        w = tf.tile(w, [shape[0], 1, 1, 1, 1, 1, 1])
        u_hat = tf.matmul(w, u_ij, transpose_a=True)
        # size of u_hat: [batch_size, channels*num_capsules, next_channels*next_num_capsules, next_vec_len, 1]
        u_hat = tf.reshape(u_hat, [shape[0], shape[1]*shape[2], -1, next_output_vector_len, 1])
        u_hat_without_backprop = tf.stop_gradient(u_hat, "u_hat_without_backprop")
        b_ij = tf.constant(np.zeros([shape[0], shape[1]*shape[2], next_num_channels*next_num_capsules, 1, 1]), dtype=tf.float32)
        c_ij = tf.nn.softmax(b_ij, dim=2)
        for r in range(num_iter):
            if r != num_iter-1:
                # size of s_j: [batch_size, 1, next_channels*next_num_capsules, next_output_vector_len, 1]
                s_j = tf.reduce_sum(tf.multiply(c_ij, u_hat_without_backprop), axis=1, keep_dims=True)
                v_j = squashing(s_j)
                v_j =tf.tile(v_j, [1, shape[1]*shape[2], 1, 1, 1])
                # b_ij += u_hat * v_j
                b_ij = b_ij + tf.matmul(u_hat, v_j, transpose_a=True)
            else:
                s_j = tf.reduce_sum(tf.multiply(c_ij, u_hat), axis=1, keep_dims=True)
                v_j = squashing(s_j)
    # size of v_j: [batch_size, 1, next_channels*next_num_capsules, next_output_vector_len, 1]
    return v_j



class CapsuleLayer(object):
    """ Capsule layer based on convolutional neural network
    """
    def __init__(self, num_capsules, num_channels, output_vector_len, layer_type='conv', vars_scope=None):
        self._num_capsules = num_capsules
        self._num_channels = num_channels
        self._output_vector_len = output_vector_len
        self._layer_type = layer_type
        self._vars_scope = vars_scope or "capsule_layer"

    @property
    def num_capsules(self):
        return self._num_capsules

    @property
    def output_vector_len(self):
        return self._output_vector_len

    def __call__(self, inputX, kernel_size, strides, num_iter, with_routing=True, padding='VALID'):
        input_shape = inputX.get_shape()
        with tf.variable_scope(self._vars_scope) as scope:
            if self._layer_type=='conv':
                # shape of conv1:  [batch, height, width, channels]
                kernel = tf.get_variable("conv_kernel", shape=[kernel_size[0], kernel_size[1], input_shape[-1],
                       self._num_channels*self._num_capsules*self._output_vector_len], dtype=tf.float32)
                conv_output = tf.nn.conv2d(inputX, kernel, strides, padding)
                shape1 = conv_output.get_shape()
                capsule_output = tf.reshape(conv_output, [shape1[0], 1, -1, self._output_vector_len, 1])
                if with_routing:
                    # routing(u, next_num_channels, next_num_capsules, next_output_vector_len, num_iter, scope=None):
                    # size of u: [batch_size, channels, num_capsules, output_vector_len]
                    capsule_output = routing(capsule_output, self._num_channels, self._num_capsules, self._output_vector_len, num_iter, scope)
                capsule_output = squashing(capsule_output)
                # size of capsule_output: [batch_size, num_capsules, num_vector_len, output_vector_len]
                capsule_output = tf.reshape(capsule_output, [input_shape[0], self._num_capsules, self._output_vector_len, self._num_channels])
            elif self._layer_type=='dnn':
                # here, we set with_routing to be True defaultly
                inputX = tf.reshape(inputX, [input_shape[0], 1, input_shape[1]*input_shape[3], input_shape[2], 1])
                capsule_output = routing(inputX, self._num_channels, self._num_capsules, self._output_vector_len, num_iter, scope)
                # size of s: [batch_size, 1, num_channels*num_capsules, output_vector_len, 1]
                capsule_output = squashing(capsule_output)
                # size of capsule_output: [batch_size, num_channels*num_capsules, output_vector_len]
                capsule_output = tf.squeeze(capsule_output, axis=[1, 4])
            else:
                capsule_output = None
        return capsule_output

class CapsuleNetwork(object):
    def __init__(self, args, maxTimeSteps):
        self.args = args
        self.maxTimeSteps = maxTimeSteps
        self.build_graph(self.args, self.maxTimeSteps)

    def build_graph(self, args, maxTimeSteps):
        self.maxTimeSteps = maxTimeSteps
        self.inputX = tf.placeholder(tf.float32,shape=[maxTimeSteps, args.batch_size, args.num_feature])

        # define tf.SparseTensor for ctc loss
        self.targetIxs = tf.placeholder(tf.int64)
        self.targetVals = tf.placeholder(tf.int32)
        self.targetShape = tf.placeholder(tf.int64)
        self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)
        self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size))

        self.config = {'name': args.model,
                           'num_layer': args.num_layer,
                           'num_hidden': args.num_hidden,
                           'num_class': args.num_class,
                           'activation': args.activation,
                           'optimizer': args.optimizer,
                           'learning rate': args.learning_rate,
                           'keep prob': args.keep_prob,
                           'batch size': args.batch_size}


        inputX = tf.reshape(self.inputX, [args.batch_size, maxTimeSteps, args.num_feature, 1])
        print(inputX.get_shape())
        with tf.variable_scope("layer_conv1"):
            # shape of kernel: [batch, in_height, in_width, in_channels]
            kernel = tf.get_variable("kernel", shape=[3, 3, 1, 16], dtype=tf.float32)
            # shape of conv1:  [batch, height, width, channels]
            conv1 = tf.nn.conv2d(inputX, kernel, (1,1,1,1), padding='VALID')

        print(conv1.get_shape())
        output = conv1
        for layer_id in range(args.num_layer):
            vars_scope = "capsule_cnn_layer_"+str(layer_id+1)
            # (self, num_capsules, num_channels, output_vector_len, layer_type='conv', vars_scope=None):
            capLayer = CapsuleLayer(4, 8, 2, layer_type='conv', vars_scope=vars_scope)
            # (self, inputX, kernel_size, strides, routing=True, padding='VALID'):
            output = capLayer(output, [2, 2], (1,1,1,1), args.num_iter)
            print(output.get_shape())

        # last dnn layer for classification
        vars_scope = "capsule_dnn_layer"
        capLayer = CapsuleLayer(8, 16, args.num_classes, layer_type='dnn', vars_scope=vars_scope)
        logits3d = capLayer(output, [3, 3], (1,1,1,1), args.num_iter)
        logits3d = tf.transpose(logits3d, perm=[1, 0, 2])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targetY, logits3d, self.seqLengths))
        self.var_op = tf.global_variables()
        self.var_trainable_op = tf.trainable_variables()
        if args.grad_clip == -1:
            # not apply gradient clipping
            self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        else:
            # apply gradient clipping
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.var_trainable_op), args.grad_clip)
            opti = tf.train.AdamOptimizer(args.learning_rate)
            self.optimizer = opti.apply_gradients(zip(grads, self.var_trainable_op))
        self.predictions = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits3d, self.seqLengths, merge_repeated=False)[0][0])
        if args.level == 'cha':
            self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))
        self.initial_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)


## test code for simplicity
if __name__ == '__main__':
    sess=tf.InteractiveSession()
    conv1 = tf.constant(np.random.rand(2,20,20,2), dtype=tf.float32)
    # (self, num_capsules, num_channels, output_vector_len, layer_type='conv', vars_scope=None):
    # (self, inputX, kernel_size, strides, routing=True, padding='VALID'):
    capLayer1 = CapsuleLayer(2, 3, 10, layer_type='conv', vars_scope="testlayer1")
    output = capLayer1(conv1, [3, 3], (1,1,1,1), 3)

    capLayer2 = CapsuleLayer(2, 3, 10, layer_type='dnn', vars_scope="testlayer2")
    output = capLayer2(output, [3, 3], (1,1,1,1), 3)

    sess.run(tf.global_variables_initializer())
    print(sess.run(output))
