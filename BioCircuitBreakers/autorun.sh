#!/bin/bash

args=(
    "-m indy -a indy_20160622_01" 
    "-m indy -a indy_20160630_01"
    "-m indy -a indy_20170131_02"
    "-m loco -a loco_20170210_03"
    "-m loco -a loco_20170215_02"
    "-m loco -a loco_20170301_05"
)

for arg in "${args[@]}"; do
    python train.py -r results_recwise/ -n AEGRU $arg
done

wait

echo "All done"
