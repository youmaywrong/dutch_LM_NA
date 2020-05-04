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
                    'that_nounpp_adv'
                    'rel_def'
                    'rel_nondef'
                    'rel_def_obj'
                    )

for task in ${templates[@]}; do
    echo Extracting predictions for $task
    python extract_predictions.py -m model.pt -i data/tasks/$task --cuda
done
