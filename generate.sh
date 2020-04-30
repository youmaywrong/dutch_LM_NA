#!/bin/bash

# python generate.py -t simple
# python generate.py -t adv
# python generate.py -t adv_adv
# python generate.py -t adv_conjunction
# python generate.py -t name_pp --prepositions_num 5 --proper_nouns 6
# python generate.py -t noun_pp --prepositions_num 3
# python generate.py -t noun_pp_adv --verbs_num 8 --position_nouns 5 --prepositions_num 3 --adverbs1_num 4
# python generate.py -t qnty_simple
# python generate.py -t rel_sg --verbs_num 8
# python generate.py -t rel_sg_adv --verbs_num 8 --adverbs1_num 4
# python generate.py -t rel_pl --verbs_num 8
# python generate.py -t rel_pl_adv --verbs_num 8 --adverbs1_num 4
python generate.py -t qnty_noun_pp --prepositions_num 3
python generate.py -t qnty_name_pp --prepositions_num 5 --proper_nouns 6
