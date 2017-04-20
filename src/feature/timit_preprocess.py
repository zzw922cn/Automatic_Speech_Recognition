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
import sklearn
from sklearn import preprocessing


## keywords can be set to either of 'train' and 'test'
level = 'cha'
keywords = 'train'

mfcc_dir = '/home/pony/github/data/timit/'+level+'/'+keywords+'/mfcc/'
label_dir = '/home/pony/github/data/timit/'+level+'/'+keywords+'/label/'

if not os.path.exists(label_dir):
    os.makedirs(label_dir)
if not os.path.exists(mfcc_dir):
    os.makedirs(mfcc_dir)

rootdir = '/media/pony/Seagate Expansion Drive/学习/语音识别/ASR数据库/TIMIT/'+keywords

count = 0
## original phonemes
phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

## cleaned phonemes
#phn = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']

for subdir, dirs, files in os.walk(rootdir):
    if True:
        for file in files:
            fullFilename = os.path.join(subdir, file)
	    filenameNoSuffix =  os.path.splitext(fullFilename)[0]
	    if file.endswith('.WAV'):
	        (rate,sig)= wav.read(fullFilename)
                mfcc = calcMFCC_delta_delta(sig,rate,win_length=0.020,win_step=0.010)
		mfcc = preprocessing.scale(mfcc)
		mfcc = np.transpose(mfcc)
		print mfcc.shape
		m_f = mfcc_dir + filenameNoSuffix.split('/')[-2]+'-'+filenameNoSuffix.split('/')[-1]+'.npy'
		np.save(m_f,mfcc)
		if level == 'phn':
		    labelFilename = filenameNoSuffix + '.PHN'
    	            phenome = []
                    with open(labelFilename,'r') as f:
		        for line in f.read().splitlines():
			    s=line.split(' ')[2]
			    p_index = phn.index(s)
			    phenome.append(p_index)
		    print phenome
		    phenome = np.array(phenome)
		elif level == 'cha':
		    labelFilename = filenameNoSuffix + '.WRD'
    	            phenome = []
		    sentence = ''
                    with open(labelFilename,'r') as f:
		        for line in f.read().splitlines():
			    s=line.split(' ')[2]
			    sentence += s+' '
			    for c in s:
				if c=="'":
				    phenome.append(27)
				else:
				    phenome.append(ord(c)-96)
			    phenome.append(0)
		    phenome = phenome[:-1]
		    print phenome
		    print sentence

		t_f = label_dir + filenameNoSuffix.split('/')[-2]+'-'+filenameNoSuffix.split('/')[-1]+'.npy'
		print t_f
		np.save(t_f,phenome)
		count+=1
		print 'file index:',count

