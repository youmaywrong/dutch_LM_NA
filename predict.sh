#!/bin/bash

declare -a templates=('simple'
                    'adv'
                    'nounpp'
                    'namepp'
                    'noun_conj'
                    's_conj'
                    'qnty_simple'
                    'qnty_nounpp'
                    'qnty_namepp'
                    'that'
                    'that_adv'
                    'that_compl'
                    'that_nounpp'
                    'rel_def'
                    'rel_nondef'
                    'rel_def_obj'
                    )

declare -a seeds=('22'
                 '23')

for s in ${seeds[@]}; do
    for task in ${templates[@]}; do
        echo Extracting predictions for $task
        python predict.py -m dutch_hidden650_batch64_dropout0.2_lr20.0_seed_$s.pt -i data/tasks/$task.tsv -o output_$s --cuda
    done
done
