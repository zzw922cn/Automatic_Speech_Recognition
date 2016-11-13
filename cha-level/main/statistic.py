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
import glob
import sys
import shutil
count=0
fi = 0
key = 'label'
mfccDir = '/home/pony/github/data/'+key+'/'
for subdir, dirs, files in os.walk(mfccDir):
    for f in files:
	if f.endswith('.npy'):
            fullFilename = os.path.join(subdir, f)
	    print 'index:',count
	    new_dir = '/home/pony/github/newdata/'+key+'/'+str(fi)+'/'
	    print new_dir
	    if os.path.exists(str(new_dir)):  
                pass  
            else:  
                os.mkdir(str(new_dir))  
	    shutil.move(fullFilename,new_dir)
	    count = count+1
	    if count%15360 == 0:
		fi = fi + 1
print count
