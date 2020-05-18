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
    echo $task
    python -W ignore ../ablation.py -i data/tasks/$task.tsv -u 873
done
