# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : nist2wav.py
# Description  : Converting nist format to wav format for Automatic Speech Recognition
# ******************************************************


import subprocess
import os

def nist2wav(src_dir):
    count = 0
    for subdir, dirs, files in os.walk(src_dir):
        for f in files:
            fullFilename = os.path.join(subdir, f)
            if f.endswith('.wv1') or f.endswith('.wv2'):
                count += 1
                os.system("./sph2pipe_v2.5/sph2pipe "+fullFilename+" -f rif " +fullFilename+".wav")
                print(fullFilename)

if __name__ == '__main__':
    nist2wav('dataset')
