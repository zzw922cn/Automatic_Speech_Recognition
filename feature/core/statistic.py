#-*- coding:utf-8 -*-
#!/usr/bin/python

''' 
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
import theano
import theano.tensor as T
from sigprocess import *
from calcmfcc import audio2fft
import scipy.io.wavfile as wav
import numpy as np
np.set_printoptions(threshold=np.nan)
import cPickle
import glob
import sys
sys.path.append('../utils/')
from functionUtils import writeParamsIntoFile,readParamsFromFile,writeBinaryArrayForC,normalize
import re

a=T.matrix()
b=normalize(a)
norm=theano.function([a],b)


mfccDir = '/home/pony/DeepVoice4.0/mfcc_and_label/TIMIT'
for subdir, dirs, files in os.walk(mfccDir):
    count=0
    for file in files:
        fullFilename = os.path.join(subdir, file)
	filenameNoSuffix =  os.path.splitext(fullFilename)[0]
	if file.endswith('.mfcc'):
	    count = count+1
    if count!=10:
	print count
