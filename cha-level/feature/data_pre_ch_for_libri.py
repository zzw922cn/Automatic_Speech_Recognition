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

rootdir = '/home/pony/ASR/datasets/LibriSpeech/train-other-500/'


count = 0
subset = 0
labels=[]

label_dir = '/home/pony/github/ASR_libri/libri/cha-level/mfcc_and_label/label/'
mfcc_dir = '/home/pony/github/ASR_libri/libri/cha-level/mfcc_and_label/mfcc/'

if True:
    for subdir, dirs, files in os.walk(rootdir):
        if True:
            for f in files:
                fullFilename = os.path.join(subdir, f)
	        filenameNoSuffix =  os.path.splitext(fullFilename)[0]
	        if f.endswith('.wav'):
		    print fullFilename
	            (rate,sig)= wav.read(fullFilename)
                    mfcc = calcMFCC_delta_delta(sig,rate,win_length=0.020,win_step=0.020)
		    # transpose mfcc to array of (39,time_length)
		    mfcc = np.transpose(mfcc)
		    print mfcc.shape
		    # save mfcc to file
		    m_f = mfcc_dir + filenameNoSuffix.split('/')[-1] +'.npy'
		    np.save(m_f,mfcc)
                    
	            labelFilename = filenameNoSuffix + '.label'
                    with open(labelFilename,'r') as f:
    	    	        characters = f.readline().strip()
	            print characters
    	            targets = []
    	            for c in characters:
			if c == ' ':
			    targets.append(0)
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
		 
