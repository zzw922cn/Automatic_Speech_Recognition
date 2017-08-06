#-*- coding:utf-8 -*-
#!/usr/bin/python
''' Automatic Speech Recognition

author(s):
zzw922cn, nemik

date:2017-4-15
'''
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append('../')

'''
Do MFCC over all *.wav files and parse label file Use os.walk to iterate all files in a root directory

original phonemes:

phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

mapped phonemes(For more details, you can read the main page of this repo):
phn = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
'''

import os
import argparse
from core.sigprocess import *
from core.calcmfcc import calcfeat_delta_delta
from core.spectrogram import spectrogramPower
import scipy.io.wavfile as wav
import numpy as np
import glob
import sys
import sklearn
from sklearn import preprocessing
from scikits.audiolab import Format, Sndfile
from scikits.audiolab import wavread

## original phonemes
phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

## cleaned phonemes
#phn = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']

def wav2feature(rootdir, save_directory, mode, feature_len, level, keywords, win_len, win_step,  seq2seq, save):
    feat_dir = os.path.join(save_directory, level, keywords, mode)
    label_dir = os.path.join(save_directory, level, keywords, 'label')
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            fullFilename = os.path.join(subdir, file)
            filenameNoSuffix =  os.path.splitext(fullFilename)[0]
            if file.endswith('.WAV'):
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
                feat = calcfeat_delta_delta(sig,rate,win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
                feat = preprocessing.scale(feat)
                feat = np.transpose(feat)
                print(feat.shape)

                if level == 'phn':
                    labelFilename = filenameNoSuffix + '.PHN'
                    phenome = []
                    with open(labelFilename,'r') as f:
                        if seq2seq is True:
                            phenome.append(len(phn)) # <start token>
                        for line in f.read().splitlines():
                            s=line.split(' ')[2]
                            p_index = phn.index(s)
                            phenome.append(p_index)
                        if seq2seq is True:
                            phenome.append(len(phn)+1) # <end token>
                        print(phenome)
                    phenome = np.array(phenome)

                elif level == 'cha':
                    labelFilename = filenameNoSuffix + '.WRD'
                    phenome = []
                    sentence = ''
                    with open(labelFilename,'r') as f:
                        for line in f.read().splitlines():
                            s=line.split(' ')[2]
                            sentence += s+' '
                            if seq2seq is True:
                                phenome.append(28)
                            for c in s:
                                if c=="'":
                                    phenome.append(27)
                                else:
                                    phenome.append(ord(c)-96)
                            phenome.append(0)

                        phenome = phenome[:-1]
                        if seq2seq is True:
                            phenome.append(29)
                    print(phenome)
                    print(sentence)

                count+=1
                print('file index:',count)
                if save:
                    featureFilename = feat_dir + filenameNoSuffix.split('/')[-2]+'-'+filenameNoSuffix.split('/')[-1]+'.npy'
                    np.save(featureFilename,feat)
                    labelFilename = label_dir + filenameNoSuffix.split('/')[-2]+'-'+filenameNoSuffix.split('/')[-1]+'.npy'
                    print(labelFilename)
                    np.save(labelFilename,phenome)


if __name__ == '__main__':
    # character or phoneme
    parser = argparse.ArgumentParser(prog='timit_preprocess',
                                     description="""
                                     Script to preprocess timit data
                                     """)
    parser.add_argument("path", help="Directory where Timit dataset is contained", type=str)
    parser.add_argument("save", help="Directory where preprocessed arrays are to be saved",
                        type=str)
    parser.add_argument("-n", "--name", help="Name of the dataset",
                        choices=['train', 'test'],
                        type=str, default='train')
    parser.add_argument("-l", "--level", help="Level",
                        choices=['cha', 'phn'],
                        type=str, default='cha')
    parser.add_argument("-m", "--mode", help="Mode",
                        choices=['mfcc', 'fbank'],
                        type=str, default='mfcc')
    parser.add_argument('--featlen', type=int, default=13, help='Features length')
    parser.add_argument("--seq2seq", help="set this flag to use seq2seq", action="store_true")

    parser.add_argument("-winlen", "--winlen", type=float,
                        default=0.02, help="specify the window length of feature")

    parser.add_argument("-winstep", "--winstep", type=float,
                        default=0.01, help="specify the window step length of feature")

    args = parser.parse_args()
    root_directory = args.path
    save_directory = args.save
    level = args.level
    mode = args.mode
    feature_len = args.featlen
    name = args.name
    seq2seq = args.seq2seq
    win_len = args.winlen
    win_step = args.winstep

    root_directory = os.path.join(root_directory, name)
    if root_directory == ".":
        root_directory = os.getcwd()
    if save_directory == ".":
        save_directory = os.getcwd()
    if not os.path.isdir(root_directory):
        raise ValueError("Root directory does not exist!")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    wav2feature(root_directory, save_directory, mode=mode, feature_len=feature_len,
                level=level, keywords=name, win_len=win_len, win_step=win_step,
                seq2seq=seq2seq, save=True)
