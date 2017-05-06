import os
import subprocess

def extract(rootdir):
  for subdir, dirs, files in os.walk(rootdir):
    for f in files:
      if f.endswith('.zip'):
        fullFilename = os.path.join(rootdir, f)
        subprocess.call(['atool', '-x', fullFilename])
        print f

extract('./')
