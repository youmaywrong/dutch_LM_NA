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
                    )

for task in ${templates[@]}; do
    python finalise_tasks.py -d full_data -t $task
done
