#!/bin/sh

for m in $(seq 50 2 100)
do
    echo $model
    echo synth_test for model $m
    python3 test.py --data test -bt 0.4 -it 0.2 -m $m
    echo real_test for model $m
    python3 test.py --data real -bt 0.4 -it 0.2 -m $m
done
