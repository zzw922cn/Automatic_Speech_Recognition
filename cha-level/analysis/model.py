#coding=utf-8
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import GRULayer,count_params,FeaturePoolLayer,Pool2DLayer,MaxPool1DLayer,MaxPool2DLayer,Conv1DLayer,Conv2DLayer,ReshapeLayer,NINLayer,InputLayer,GaussianNoiseLayer,DropoutLayer,DenseLayer,LSTMLayer,RecurrentLayer,ElemwiseSumLayer,DimshuffleLayer,get_all_layers,ConcatLayer
from lasagne.layers.shape import FlattenLayer,ReshapeLayer
from lasagne.regularization import regularize_layer_params,l1,l2
from lasagne.nonlinearities import linear,sigmoid,tanh,rectify,leaky_rectify,elu,softmax
from lasagne.layers import LocalResponseNormalization2DLayer,BatchNormLayer,batch_norm
import config

def build_Deep2DCnnDnn_1(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],-1,[2]))
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = ReshapeLayer(l_in,(1,1,-1,[2]))

    base = 16
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    end = conv12
    bs,ch,ro,co = end.output_shape
    network = ReshapeLayer(end,(-1,co*ch))
    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network

def build_Deep2DCnnDnn_2(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],-1,[2]))
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = ReshapeLayer(l_in,(1,1,-1,[2]))

    base = 24
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/3,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/3,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    end = conv12
    bs,ch,ro,co = end.output_shape
    network = ReshapeLayer(end,(-1,co*ch))
    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network

def build_Deep2DCnnDnn_19(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],-1,[2]))
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = ReshapeLayer(l_in,(1,1,-1,[2]))

    base = 24
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base*2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base*2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    bs,ch,ro,co = conv12.output_shape
    network = ReshapeLayer(conv12,(-1,co*ch))
    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network

def build_Deep2DCnnDnn_20(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],-1,[2]))
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = ReshapeLayer(l_in,(1,1,-1,[2]))

    base = 16
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    bs,ch,ro,co = conv12.output_shape
    network = ReshapeLayer(conv12,(-1,co*ch))
    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network

def build_Deep2DCnnDnn_26(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],-1,[2]))
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in ,p=config.dropout)
    l_in = ReshapeLayer(l_in,(1,1,-1,[2]))

    base = 24
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/16,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/16,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    bs,ch,ro,co = conv12.output_shape
    network = ReshapeLayer(conv12,(-1,co*ch))
    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network

def build_Deep2DCnnDnn_27(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],-1,[2]))
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = ReshapeLayer(l_in,(1,1,-1,[2]))

    base = 32
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base*2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base*2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv13 = Conv2DLayer(conv12,num_filters=base/16,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv14 = Conv2DLayer(conv13,num_filters=base/16,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    bs,ch,ro,co = conv14.output_shape
    network = ReshapeLayer(conv14,(-1,co*ch))
    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network

def build_Deep2DCnnDnn_28(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],-1,[2]))
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in ,p=config.dropout)
    l_in = ReshapeLayer(l_in,(1,1,-1,[2]))

    base = 16
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    bs,ch,ro,co = conv12.output_shape
    network = ReshapeLayer(conv12,(-1,co*ch))
    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network


def build_Deep2DCnnDnn_29(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],-1,[2]))
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in ,p=config.dropout)
    l_in = ReshapeLayer(l_in,(1,1,-1,[2]))

    base = 16
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    bs,ch,ro,co = conv12.output_shape
    network = ReshapeLayer(conv12,(-1,co*ch))
    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network

def build_Deep2DCnnDnn_30(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],1,-1,[2]))

    base = 16
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    bs,ch,ro,co = conv12.output_shape
    l_in = ReshapeLayer(conv12,(1,-1,co*ch))

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    network = ReshapeLayer(l_in,(-1,[2]))

    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network

def build_Deep2DCnnDnn_31(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],1,-1,[2]))

    base = 24
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/12,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/12,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    bs,ch,ro,co = conv12.output_shape
    l_in = ReshapeLayer(conv12,(1,-1,co*ch))

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    network = ReshapeLayer(l_in,(-1,[2]))

    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network

def build_Deep2DCnnDnn_34(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],1,-1,[2]))

    base = 32
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv11 = Conv2DLayer(conv10,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv13 = Conv2DLayer(conv12,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv14 = Conv2DLayer(conv13,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv15 = Conv2DLayer(conv14,num_filters=base/16,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv16 = Conv2DLayer(conv15,num_filters=base/16,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    end = conv16

    bs,ch,ro,co = end.output_shape
    l_in = ReshapeLayer(end,(1,-1,co*ch))

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    network = ReshapeLayer(l_in,(-1,[2]))

    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network


def build_Deep2DCnnDnn_37(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],1,-1,[2]))

    base = 16
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')


    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')


    conv9 = Conv2DLayer(conv8,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv11 = Conv2DLayer(conv10,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')


    bs,ch,ro,co = conv12.output_shape
    l_in = ReshapeLayer(conv12,(1,-1,co*ch))

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    network = ReshapeLayer(l_in,(-1,[2]))

    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network


def build_Deep2DCnnDnn_38(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],1,-1,[2]))

    base = 16
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')


    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv11 = Conv2DLayer(conv10,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv13 = Conv2DLayer(conv12,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv14 = Conv2DLayer(conv13,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv15 = Conv2DLayer(conv14,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv16 = Conv2DLayer(conv15,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    end = conv16
    bs,ch,ro,co = end.output_shape
    l_in = ReshapeLayer(end,(1,-1,co*ch))

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    network = ReshapeLayer(l_in,(-1,[2]))

    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network


def build_Deep2DCnnDnn_r30(input_dim=39,input_var=None):

    '''Res of model 30'''

    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],1,-1,[2]))

    base = 16
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv10 = ElemwiseSumLayer([conv1,conv10])

    conv11 = Conv2DLayer(conv10,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv12 = ElemwiseSumLayer([conv11,conv12])

    bs,ch,ro,co = conv12.output_shape
    l_in = ReshapeLayer(conv12,(1,-1,co*ch))

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    network = ReshapeLayer(l_in,(-1,[2]))

    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network

def build_Deep2DCnnDnn_r31(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],1,-1,[2]))

    base = 24
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    
    conv10 = ElemwiseSumLayer([conv1,conv10])

    conv11 = Conv2DLayer(conv10,num_filters=base/12,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/12,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv12 = ElemwiseSumLayer([conv11,conv12])

    bs,ch,ro,co = conv12.output_shape
    l_in = ReshapeLayer(conv12,(1,-1,co*ch))

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    network = ReshapeLayer(l_in,(-1,[2]))

    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network


def build_Deep2DCnnDnn_r34(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],1,-1,[2]))

    base = 32
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv7 = Conv2DLayer(conv6,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv9 = Conv2DLayer(conv8,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv10 = ElemwiseSumLayer([conv1,conv10])

    conv11 = Conv2DLayer(conv10,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv12 = ElemwiseSumLayer([conv11,conv12])

    conv13 = Conv2DLayer(conv12,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv14 = Conv2DLayer(conv13,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv14 = ElemwiseSumLayer([conv13,conv14])

    conv15 = Conv2DLayer(conv14,num_filters=base/16,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv16 = Conv2DLayer(conv15,num_filters=base/16,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv16 = ElemwiseSumLayer([conv15,conv16])

    end = conv16

    bs,ch,ro,co = end.output_shape
    l_in = ReshapeLayer(end,(1,-1,co*ch))

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    network = ReshapeLayer(l_in,(-1,[2]))

    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network


def build_Deep2DCnnDnn_r37(input_dim=39,input_var=None):
    # input shape(batch_size,time length, dimension)
    l_in = InputLayer(shape=(1,None,input_dim),input_var = input_var)
    l_in = GaussianNoiseLayer(l_in,sigma=0.1)
    l_in = ReshapeLayer(l_in,([0],1,-1,[2]))

    base = 16
    conv1 = Conv2DLayer(l_in,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv2 = Conv2DLayer(conv1,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv3 = Conv2DLayer(conv2,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv4 = Conv2DLayer(conv3,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv5 = Conv2DLayer(conv4,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv6 = Conv2DLayer(conv5,num_filters=base,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv6 = ElemwiseSumLayer([conv1,conv6])

    conv7 = Conv2DLayer(conv6,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv8 = Conv2DLayer(conv7,num_filters=base/2,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv8 = ElemwiseSumLayer([conv7,conv8])

    conv9 = Conv2DLayer(conv8,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv10 = Conv2DLayer(conv9,num_filters=base/4,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv10 = ElemwiseSumLayer([conv9,conv10])

    conv11 = Conv2DLayer(conv10,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')
    conv12 = Conv2DLayer(conv11,num_filters=base/8,filter_size=(3,3),stride=(1,1),nonlinearity=elu,pad='same')

    conv12 = ElemwiseSumLayer([conv11,conv12])

    bs,ch,ro,co = conv12.output_shape
    l_in = ReshapeLayer(conv12,(1,-1,co*ch))

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = RecurrentLayer(l_in, num_units=128, nonlinearity = elu)
    l_in = DropoutLayer(l_in,p=config.dropout)

    network = ReshapeLayer(l_in,(-1,[2]))

    network = DenseLayer(network, num_units=256, nonlinearity = elu)
    network = DropoutLayer(network,p=config.dropout)
    network = DenseLayer(network, num_units=62, nonlinearity = elu)
    network = ReshapeLayer(network,([0],1,[1]))
    return network


