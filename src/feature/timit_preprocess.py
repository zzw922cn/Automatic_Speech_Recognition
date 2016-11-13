#-*- coding:utf-8 -*-
#!/usr/bin/python

''' Do MFCC over all *.wav files and parse label file Use os.walk to iterate all files in a root directory
Ascii table(Total 61 phonemes ):

phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

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
import glob
import sys

mfcc_dir = '/home/pony//github/data/timit/test/mfcc/'
label_dir = '/home/pony//github/data/timit/test/label/'

rootdir = '/home/pony/ASR/datasets/TIMIT/TEST/'

count = 0
phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

for subdir, dirs, files in os.walk(rootdir):
    if True:
        for file in files:
            fullFilename = os.path.join(subdir, file)
	    filenameNoSuffix =  os.path.splitext(fullFilename)[0]
	    if file.endswith('.WAV'):
	        (rate,sig)= wav.read(fullFilename)
                mfcc = calcMFCC_delta_delta(sig,rate,win_length=0.020,win_step=0.010)
		mfcc = np.transpose(mfcc)
		print mfcc.shape
		m_f = mfcc_dir + filenameNoSuffix.split('/')[-2]+'-'+filenameNoSuffix.split('/')[-1]+'.npy'
		np.save(m_f,mfcc)

		labelFilename = filenameNoSuffix + '.PHN'
    	        phenome = []
                with open(labelFilename,'r') as f:
		    for line in f.read().splitlines():
			s=line.split(' ')[2]
			p_index = phn.index(s)
			phenome.append(p_index)
		print phenome
		phenome = np.array(phenome)
		t_f = label_dir + filenameNoSuffix.split('/')[-2]+'-'+filenameNoSuffix.split('/')[-1]+'.npy'
		print t_f
		np.save(t_f,phenome)
		count+=1
		print 'file index:',count

