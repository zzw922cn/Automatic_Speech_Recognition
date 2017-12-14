# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : rename_wsj.py
# Description  : Renaming some folders for WSJ dataset
# ******************************************************

import subprocess
import os

def lookup(cd_id, logfile):
  # find new name according to indexing
  with open(logfile, 'r') as f:
    content = f.readlines()
  for line in content:
    if int(line.split(' ')[-1][2:]) == int(cd_id[2:]):
      if '.' in line.split(' ')[-3]:
        newName = line.split(' ')[-3]
        return newName
    else:
      continue

def renameCD(src_dir, mode):
  # rename CD directory to new name
  logfile = mode+'.links.log'
  cd_dir = os.path.join(src_dir, mode)
  count = 0
  for subdir in os.listdir(cd_dir):
    if subdir.startswith('CD') or subdir.startswith('cd'):
      newName = lookup(subdir, os.path.join(src_dir, logfile))
      cd_path = os.path.join(src_dir, mode, subdir)
      new_cd_path = os.path.join(src_dir, mode, newName)
      os.rename(cd_path, new_cd_path)
      count += 1
      print('new file ', count, ': ', new_cd_path)

if __name__ == '__main__':
  renameCD('/media/pony/DLdigest/study/ASR/corpus/wsj', mode='wsj0')
  ## you should add cd34, cd16 to the wsj1.links.log file and then execute this command
  renameCD('/media/pony/DLdigest/study/ASR/corpus/wsj', mode='wsj1')
