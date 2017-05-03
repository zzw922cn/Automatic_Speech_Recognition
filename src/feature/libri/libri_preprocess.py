#-*- coding:utf-8 -*-
#!/usr/bin/python
''' Automatic Speech Recognition

author:
zzw922cn
     
date:2017-4-15

Preprocessing for LibriSpeech corpus
'''

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

from utils.sigprocess import *
from utils.calcmfcc import calcMFCC_delta_delta
import scipy.io.wavfile as wav
import numpy as np
import os
import cPickle
import glob
import sklearn
from sklearn import preprocessing

def wav2feature(rootdir, mfcc_dir, label_dir, win_len=0.02, win_step=0.01, mode='mfcc', keyword='dev-clean', seq2seq=False, save=False):
  count = 0
  for subdir, dirs, files in os.walk(rootdir):
    for f in files:
      fullFilename = os.path.join(subdir, f)
      filenameNoSuffix =  os.path.splitext(fullFilename)[0]
      if f.endswith('.wav'):
        rate = None
        sig = None
        try:
          (rate,sig)= wav.read(fullFilename)
        except ValueError as e:
          if e.message == "File format 'NIST'... not understood.":
            sf = Sndfile(fullFilename, 'r')
            nframes = sf.nframes
            sig = sf.read_frames(nframes)
            rate = sf.samplerate
        mfcc = calcMFCC_delta_delta(sig,rate,win_length=win_len,win_step=win_step)
        mfcc = preprocessing.scale(mfcc)
        mfcc = np.transpose(mfcc)
        print mfcc.shape
        labelFilename = filenameNoSuffix + '.label'
        with open(labelFilename,'r') as f:
          characters = f.readline().strip().lower()
        targets = []
        if seq2seq is True:
          targets.append(28)
        for c in characters:
          if c == ' ':
            targets.append(0)
          elif c == "'":
            targets.append(27)
          else:
            targets.append(ord(c)-96) #从1开始
        if seq2seq is True:
          targets.append(29)
        targets = np.array(targets)
        print targets
        count+=1
        print 'file index:',count
        if save:
          featureFilename = mfcc_dir + filenameNoSuffix.split('/')[-1] +'.npy'
          np.save(featureFilename,mfcc)
          t_f = label_dir + filenameNoSuffix.split('/')[-1] +'.npy'
          print t_f
          np.save(t_f,targets)
          
         
if __name__ == '__main__':
  keywords = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']
  keyword = keywords[0]
  label_dir = '/home/pony/github/data/libri/cha-level/'+keyword+'/label/'
  mfcc_dir = '/home/pony/github/data/libri/cha-level/'+keyword+'/mfcc/'
  if not os.path.exists(label_dir):
    os.makedirs(label_dir)
  if not os.path.exists(mfcc_dir):
    os.makedirs(mfcc_dir)

  rootdir = '/media/pony/DLdigest/study/ASR/corpus/LibriSpeech/'+keyword
  wav2feature(rootdir, mfcc_dir, label_dir, win_len=0.02, win_step=0.01, mode='mfcc', keyword=keyword, seq2seq=True, save=False)
