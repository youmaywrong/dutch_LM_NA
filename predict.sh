#!/bin/bash

declare -a templates=('adv'
                    'namepp'
                    'noun_conj'
                    'nounpp'
                    'qnty_namepp'
                    'qnty_nounpp'
                    'qnty_simple'
                    's_conj'
                    'simple'
                    'that'
                    'that_adv'
                    'that_compl'
                    'that_nounpp'
                    'rel_def'
                    'rel_nondef'
                    'rel_def_obj'
                    )

for task in ${templates[@]}; do
    echo Extracting predictions for $task
    python predict.py -m model.pt -i data/tasks/$task.tsv --cuda
done
