#coding=utf-8
''' convert .flac files of LibriSpeech to .wav files

2016-9-5
Zhang Zewang
'''

import os

count = 0
ROOTDIR='/media/pony/Seagate Expansion Drive/学习/语音识别/ASR数据库/LibriSpeech/'
for subdir, dirs, files in os.walk(ROOTDIR):
    print 'len:',len(files)
    for f in files:
	fullFilename = os.path.join(subdir, f)
	filenameNoSuffix =  os.path.splitext(fullFilename)[0]
	if f.endswith('.wav'):
	    count = count+1
	if f.endswith('.TXT'):
	    os.remove(fullFilename)
	elif f.endswith('.flac'):
	    print fullFilename
	    os.remove(fullFilename)
	elif f.endswith('.txt'):
	    count = count+1
	    with open(fullFilename) as ff:
	        lines = ff.readlines()
	    for line in lines:
		sub_n = line.split(' ')[0]+'.label'
		sub_file = os.path.join(subdir,sub_n)

		sub_c = ' '.join(line.split(' ')[1:])
		## keep blank, keep '
		sub_c = sub_c.lower()
		print sub_c
		with open(sub_file,'w') as sf:
		    sf.write(sub_c)
		    print sub_file
		print 'file num:', count
	
print 'in all:',count
	    
