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
import scipy.io.wavfile as wav
rootdir='./205/'
for subdir, dirs, files in os.walk(rootdir):
    if len(files)!=40:
        for file in files:
            fullFilename = os.path.join(subdir, file)
	    filenameNoSuffix =  os.path.splitext(fullFilename)[0]
	    if file.endswith('.wav'):
		print file
	        (rate,sig)= wav.read(fullFilename)
		print len(sig)
		print sig
