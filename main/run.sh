#!/bin/bash

for loop in {3..30} 
do
    echo "loop is $loop"
    b=$(( $loop % 3 ))
    echo "dataset is index $b"
    if [ $loop -eq 1 ]
    then
        python train.py --lb=$b || break
    else
        python train.py --lb=$b --keep=True || break
    python train.py --mode=test || break
    fi
done
