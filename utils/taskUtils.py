#-*- coding:utf-8 -*-
#!/usr/bin/python
''' Automatic Speech Recognition

author(s):
zzw922cn
         
date:2017-4-15
'''

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import os

def get_num_classes(level):
    if level == 'phn':
        num_classes = 62
    elif level == 'cha':
        num_classes = 29
    elif level == 'seq2seq':
        num_classes = 30
    else:
        raise ValueError('level must be phn, cha or seq2seq, but the given level is %s'%str(level))
    return num_classes


def check_path_exists(path):
    """ check a path exists or not
    """
    if isinstance(path, list):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
    else:
        if not os.path.exists(path):
            os.makrdirs(path)

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
