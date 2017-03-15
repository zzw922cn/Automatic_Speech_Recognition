#-*- coding:utf-8 -*-
#!/usr/bin/python

''' Do MFCC over all *.wav files and parse label file Use os.walk to iterate all files in a root directory
Ascii table(Total 26 characters from 'a' to 'z'):
	

author:

      iiiiiiiiiiii            iiiiiiiiiiii         !!!!!!!             !!!!!!    
      #        ###            #        ###           ###        I#        #:     
      #      ###              #      I##;             ##;       ##       ##      
            ###                     ###               !##      ####      #       
           ###                     ###                 ###    ## ###    #'       
         !##;                    `##%                   ##;  ##   ###  ##        
        ###                     ###                     $## `#     ##  #         
       ###        #            ###        #              ####      ####;         
     `###        -#           ###        `#               ###       ###          
     ##############          ##############               `#         #     
     
date:2016-11-09
'''


import os
from sigprocess import *
from calcmfcc import calcMFCC_delta_delta
import scipy.io.wavfile as wav
import numpy as np
import cPickle
import glob
import sys
import sklearn
from sklearn import preprocessing


count = 0
#subset = 0
#labels=[]

keywords = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']

keyword = keywords[0]
label_dir = '/home/pony/github/data/libri/cha-level/'+keyword+'/label/'
mfcc_dir = '/home/pony/github/data/libri/cha-level/'+keyword+'/mfcc/'
if not os.path.exists(label_dir):
    os.makedirs(label_dir)
if not os.path.exists(mfcc_dir):
    os.makedirs(mfcc_dir)

rootdir = '/media/pony/Seagate Expansion Drive/学习/语音识别/ASR数据库/LibriSpeech/'+keyword

if True:
    for subdir, dirs, files in os.walk(rootdir):
        if True:
            for f in files:
                fullFilename = os.path.join(subdir, f)
	        filenameNoSuffix =  os.path.splitext(fullFilename)[0]
	        if f.endswith('.wav'):
		    print fullFilename
	            (rate,sig)= wav.read(fullFilename)
                    mfcc = calcMFCC_delta_delta(sig,rate,win_length=0.020,win_step=0.010)
		    mfcc = preprocessing.scale(mfcc)
		    # transpose mfcc to array of (39,time_length)
		    mfcc = np.transpose(mfcc)
		    # save mfcc to file
		    m_f = mfcc_dir + filenameNoSuffix.split('/')[-1] +'.npy'
		    np.save(m_f,mfcc)
                    
	            labelFilename = filenameNoSuffix + '.label'
                    with open(labelFilename,'r') as f:
    	    	        characters = f.readline().strip().lower()
    	            targets = []
		    ## totally 28 real characters
    	            for c in characters:
			if c == ' ':
			    targets.append(0)
			elif c == "'":
			    targets.append(27)
			else:
			    targets.append(ord(c)-96) #从1开始
		    targets = np.array(targets)
		    print targets
		    t_f = label_dir + filenameNoSuffix.split('/')[-1] +'.npy'
		    print t_f
		    # save label to file
		    np.save(t_f,targets)
		    count+=1
		    print 'file index:',count
		 
