#!/bin/bash

batch_size=64
for loop in {1..1000} 
do
    echo "loop index: $loop"
    if [ $loop -eq 1 ]
    then
        python libri_train.py --batch_size=$batch_size --mode=train || break
    else
        python libri_train.py --mode=dev --keep=True || break
        python libri_train.py --batch_size=$batch_size --mode=train --keep=True || break
    fi
done
