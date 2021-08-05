# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : split_data_by_s5.py
# Description  : Splitting data by s5 recipe
# ******************************************************

"""
NOTE:
we process the data using Kaldi s5 recipe
train set: si284
validation set: eval92
test set: dev93
"""

import shutil
import os
from speechvalley.utils import check_path_exists

def split_data_by_s5(src_dir, des_dir, keywords=['train_si284', 'test_eval92', 'test_dev93']):
  count = 0
  for key in keywords:
    wav_file_list = os.path.join(src_dir, key+'.flist') 
    label_file_list = os.path.join(src_dir, key+'.txt') 
    new_path = check_path_exists(os.path.join(des_dir, key))

    with open(wav_file_list, 'r') as wfl:
      wfl_contents = wfl.readlines()
      for line in wfl_contents:
        line = line.strip()
        if os.path.isfile(line):
          shutil.copyfile(line, os.path.join(des_dir, key, line.split('/')[-1]))
          print(line)
        else:
          tmp = '/'.join(line.split('/')[:-1]+[line.split('/')[-1].upper()])
          shutil.copyfile(tmp, os.path.join(des_dir, key, line.split('/')[-1].replace('WV1', 'wv1')))
          print(tmp)

    with open(label_file_list, 'r') as lfl:
      lfl_contents = lfl.readlines()
      for line in lfl_contents:
        label = ' '.join(line.strip().split(' ')[1:])
        with open(os.path.join(des_dir, key, line.strip().split(' ')[0]+'.label'), 'w') as lf:
          lf.writelines(label)
        print(key, label)
        

if __name__ == '__main__':
  src_dir = '/media/pony/DLdigest/study/ASR/corpus/wsj/s5/data'
  des_dir = '/media/pony/DLdigest/study/ASR/corpus/wsj/standard'
  split_data_by_s5(src_dir, des_dir)

