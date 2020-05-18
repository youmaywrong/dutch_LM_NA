#!/bin/bash
END=1300

for ((i=0;i<END;i++)); do
    echo Ablation unit $i
    python ../ablation.py -i data/tasks/nounpp.tsv -u $i --cuda
done
