#!/bin/bash

for loop in {3..30} 
do
    echo "loop is $loop"
    b=$(( $loop % 3 ))
    echo "dataset is index $b"
    if [ $loop -eq 1 ]
    then
        /usr/bin/python train.py --lb=$b || break
    else
        /usr/bin/python train.py --lb=$b --keep=True ||break
    /usr/bin/python train.py --mode=test
    fi
done
