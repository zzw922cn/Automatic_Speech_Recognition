#!/usr/bin/python

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

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
                print fullFilename

nist2wav('/home/pony/wsj/')
