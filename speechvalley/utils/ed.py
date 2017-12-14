# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : ed.py
# Description  : Calculating edit distance for Automatic Speech Recognition
# ******************************************************

import tensorflow as tf
import numpy as np

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
