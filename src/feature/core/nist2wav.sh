#!/bin/bash

# author:Rongyi Lan
# email:brianlanbo@gmail.com

# This file is designed for converting NIST format audio
# to WAV format audio, to run this script, you should install
# libsndfile software first.


target_dir=$1

fnames=(`find $target_dir -name "*.wv1"`)

for fname in "${fnames[@]}"
do
  mv "$fname" "${fname%.wav}.nist"
  sndfile-convert "${fname%.wav}.nist" "$fname"
  if [ $? = 0 ]; then
    echo renamed $fname to nist and converted back to wav using sndfile-convert
  else
    mv "${fname%.wav}.nist" "$fname"
  fi
done
