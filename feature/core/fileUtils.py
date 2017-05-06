#-*- coding:utf-8 -*-
#!/usr/bin/python
''' Automatic Speech Recognition

author(s):
zzw922cn
     
date:2017-5-5
'''

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import os

def check_path_exists(path):
  """ check a path exists or not
  """
  if not os.path.exists(path):
    os.makedirs(path)
