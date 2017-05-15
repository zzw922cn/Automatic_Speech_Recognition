#!/bin/bash

for loop in {2..300} 
do
    echo "loop index: $loop"
    if [ $loop -eq 1 ]
    then
        python libri_train.py --mode=train || break
    else
        python libri_train.py --mode=dev --keep=True || break
        python libri_train.py --batch_size=64 --mode=train --keep=True || break
    fi
done
