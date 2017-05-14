#!/bin/bash

for loop in {2..300} 
do
    echo "loop index: $loop"
    if [ $loop -eq 1 ]
    then
        python timit_train.py --mode=train || break
    else
        python timit_train.py --batch_size=1 --mode=test --keep=True || break
        python timit_train.py --batch_size=8 --mode=train --keep=True || break
    fi
done
