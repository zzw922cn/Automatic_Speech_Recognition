#!/bin/bash

for loop in {1..300} 
do
    echo "loop index: $loop"
    if [ $loop -eq 1 ]
    then
        python train.py --mode=train || break
    else
        python train.py --mode=train --keep=True || break
    python train.py --mode=test --keep=True || break
    fi
done
