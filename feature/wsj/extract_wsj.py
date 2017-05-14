#-*- coding:utf-8 -*-
#!/usr/bin/python
''' Automatic Speech Recognition

author:
zzw922cn
     
date:2017-5-5
'''

from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import os
import subprocess

def extract(rootdir):
  for subdir, dirs, files in os.walk(rootdir):
    for f in files:
      if f.endswith('.zip'):
        fullFilename = os.path.join(rootdir, f)
        subprocess.call(['atool', '-x', fullFilename])
        print f

