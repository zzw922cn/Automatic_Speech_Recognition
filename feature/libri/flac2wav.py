#coding=utf-8
''' convert .flac files of LibriSpeech to .wav files

2016-9-5
Zhang Zewang
'''
#find ./ -type f -name '*.flac' -exec flac -d {} \;
#find /opt/lampp/htdocs -type d -exec chmod 755 {} \;
#find /opt/lampp/htdocs -type f -exec chmod 644 {} \;


import os
from subprocess import call

count = 0
ROOTDIR='/home/pony/ASR/datasets/LibriSpeech/train-clean-360'
for subdir, dirs, files in os.walk(ROOTDIR):
    for f in files:
	fullFilename = os.path.join(subdir, f)
	print f
	if f.endswith('.flac'):
	    call(['flac','-d',fullFilename])
		
print 'in all:',count
	   
