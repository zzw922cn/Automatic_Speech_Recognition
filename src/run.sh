#!/bin/bash

for loop in {2..30} 
do
    echo "loop is $loop"
    if [ $loop -eq 1 ]
    then
        /usr/bin/python train.py --mode=train
    else
        /usr/bin/python train.py --mode=train --keep=True
    fi
    /usr/bin/python train.py --mode=test
done
