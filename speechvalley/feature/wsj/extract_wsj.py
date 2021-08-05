# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : extract_wsj.py
# Description  : Extracting WSJ dataset
# ******************************************************


import os
import subprocess

def extract(rootdir):
  for subdir, dirs, files in os.walk(rootdir):
    for f in files:
      if f.endswith('.zip'):
        fullFilename = os.path.join(rootdir, f)
        subprocess.call(['atool', '-x', fullFilename])
        print(f)

