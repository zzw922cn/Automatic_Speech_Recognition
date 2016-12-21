#!/usr/bin/python
#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
## 改写list_to_sparse_tensor函数使得其变成统一的编码 ##
# 原始的音素如下面的phn变量所示，但是在测试的时候，为
# 了提高性能表现，一些音素需要进行合并，具体的合并办法
# 在截图中，请根据合并方法，使得targetList变成只包含合
# 并之后的音素，可以进行重新编码，如原来是0...61，现在
# 可以变成0...47，目的是计算两个字符串之间的编辑距离。
# 并且需要改写一下get_edit_distance，使得其函数签名变成
# 如下形式：
#	get_edit_distance(hyp_arr,truth_arr,mode='train')
# mode可以取值'train'或'test'，如果是'train'模式，那么
# 就使用原始的list_to_sparse_tensor函数，如果是'test'模
# 式，那么就使用你改写后的list_to_sparse_tensor函数，换
# 句话说，你把list_to_sparse_tensor函数的签名也写成如下
# 形式：
#	list_to_sparse_tensor(targetList,mode='train')
# 这样就可以在内部设置两种模式的转换。


phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h',
       'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl',
       'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',
       'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#',
       'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',
       'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',
       'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
       'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',
       'v', 'w', 'y', 'z', 'zh']

mapping = {'ux':'uw','axr':'er','em':'m','nx':'en','n':'en',
              'eng':'ng','hv':'hh','cl':'sil','bcl':'sil','dcl':'sil',
              'gcl':'sil','epi':'sil','h#':'sil','kcl':'sil','pau':'sil',
              'pcl':'sil','tcl':'sil','vcl':'sil','l':'el','zh':'sh',
              'aa':'ao','ix':'ih','ax':'ah'}

def group_phoneme(orig_phn,mapping):
    group_phn = []
    for val in orig_phn:
        group_phn.append(val)
    group_phn.append('sil')
    for key in mapping.keys():
        if key in orig_phn:
            group_phn.remove(key)
    group_phn.sort()
    return group_phn

def list_to_sparse_tensor(targetList,mode='train'):
    ''' turn 2-D List to SparseTensor
    '''
    # NOTE: 'sil' is a new phoneme, you should care this.

    indices = [] #index
    vals = [] #value
    group_phn = group_phoneme(phn,mapping)
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            if(mode == 'train'):
                indices.append([tI, seqI])
                vals.append(val)
            elif(mode == 'test'):
                if(phn[val] in mapping.keys()):
                    val = group_phn.index(mapping[phn[val]])
                indices.append([tI, seqI])
                vals.append(val)
            else:
                raise ValueError("Invalid mode.",mode)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1] #shape
    return (np.array(indices), np.array(vals), np.array(shape))

def get_edit_distance(hyp_arr,truth_arr,mode='train'):
    ''' calculate edit distance
    '''
    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.edit_distance(hyp, truth, normalize=True)

    with tf.Session(graph=graph) as session:
        truthTest = list_to_sparse_tensor(truth_arr, mode)
        hypTest = list_to_sparse_tensor(hyp_arr, mode)
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run(editDist, feed_dict=feedDict)
    return dist

if __name__ == '__main__':
    a=[[0,5,49]]	
    b=[[21,5,10]]
    print(get_edit_distance(a,b,mode='test'))
    print(len(phn))
    print(len(mapping))
