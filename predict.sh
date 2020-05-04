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
                    
for task in ${templates[@]}; do
    echo Extracting predictions for $task
    python predict.py -m model.pt -i data/tasks/$task.tsv --cuda
done
