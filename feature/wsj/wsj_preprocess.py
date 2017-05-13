#-*- coding:utf-8 -*-
#!/usr/bin/python
''' Automatic Speech Recognition

author:
zzw922cn

date:2017-5-5
'''

from __future__ import print_function
#from __future__ import unicode_literals

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

from core.sigprocess import *
from core.calcmfcc import calcfeat_delta_delta
import scipy.io.wavfile as wav
import numpy as np
import os
import cPickle
import glob
import sklearn
from sklearn import preprocessing
from subprocess import check_call, CalledProcessError


def wav2feature(rootdir, save_dir, win_len=0.02, win_step=0.01, mode='mfcc', feature_len=feature_len, keyword='dev-clean', level='seq2seq', save=False):
  """
  To run for WSJ corpus, you should download sph2pipe_v2.5 first!
  """
  feat_dir = os.path.join(save_dir, keyword, mode)
  label_dir = os.path.join(save_dir, keyword, 'label')
  if not os.path.exists(label_dir):
    os.makedirs(label_dir)
  if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)

  count = 0
  for subdir, dirs, files in os.walk(rootdir):
    for f in files:
      fullFilename = os.path.join(subdir, f)
      filenameNoSuffix =  os.path.splitext(fullFilename)[0]
      if f.endswith('.wv1') or f.endswith('.wav'):
        rate = None
        sig = None
        try:
          (rate,sig)= wav.read(fullFilename)
        except ValueError as e:
          sph2pipe = os.path.join(sph2pipe_dir, 'sph2pipe')
          wav_name = fullFilename.replace('wv1', 'wav')
          check_call(['./sph2pipe', '-f', 'rif', fullFilename, wav_name])
          os.remove(fullFilename)
          print(wav_name)
          (rate,sig)= wav.read(wav_name)
          os.remove(fullFilename)

        feat = calcfeat_delta_delta(sig,rate,win_length=win_len,win_step=win_step)
        feat = preprocessing.scale(feat)
        feat = np.transpose(feat)
        print(feat.shape)
        labelFilename = filenameNoSuffix + '.label'
        with open(labelFilename,'r') as f:
          characters = f.readline().strip().lower()
        targets = []
        if level=='seq2seq':
          targets.append(28)
        for c in characters:
          if c == ' ':
            targets.append(0)
          elif c == "'":
            targets.append(27)
          else:
            targets.append(ord(c)-96)
        if level=='seq2seq':
          targets.append(29)
        targets = np.array(targets)
        print(targets)
        count+=1
        print('file index:', count)
        if save:
          featureFilename = os.path.join(feat_dir, filenameNoSuffix.split('/')[-1] +'.npy')
          np.save(featureFilename,feat)
          t_f = os.path.join(label_dir, filenameNoSuffix.split('/')[-1] +'.npy')
          print(t_f)
          np.save(t_f,targets)

if __name__ == '__main__':
    keywords = ['train_si284', 'test_eval92', 'test_dev93']
    for keyword in keywords:
        rootdir = os.path.join('/media/pony/DLdigest/study/ASR/corpus/wsj/standard', keyword)
        wav2feature(rootdir, save_dir='/home/pony/github/data/wsj', win_len=0.02, win_step=0.01, mode='mfcc', feature_len=feature_len, keyword=keyword, level='cha', save=True)
