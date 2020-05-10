import os, sys
import csv
import random
import argparse

from collections import defaultdict
from nltk.parse.generate import generate
from tqdm import tqdm

from grammar import get_grammar, get_grammar_string

def read_words(filename, n=-1):
    """
    Read vocabulary terms from a csv file (delimiter is a comma)
    The headers of the columns indicate the number if applicable, the
    part-of-speech tag otherwise.
    Example 1: singular, plural
    Example 2: adverb
    Args:
        filename (str): csv filename
        n (int): number of rows to sample to avoid combinatorial explosion
    Returns:
        dict: structure of {column_name: string formatted for nltk grammars}
    """
    columns = defaultdict(list)
    reader = list(csv.DictReader(open(filename), delimiter=","))
    n = len(reader) if n == -1 else min(n, len(reader))
    reader = random.sample(reader, n)

    for row in reader:
        for (k, v) in row.items():
            columns[k].append(f"'{v}'")

    return {k: " | ".join(v) for k, v in columns.items()}


def generate_dataset(grammar, correct, incorrect):
    """
    Generate data with correct and incorrect number-verb agreement.

    Args:
        grammar (str): NLTK feature grammar
        correct (dict): for each number condition (key) a start symbol rule
                        (value) to create sentences with noun-verb agreement
        incorrect (dict): for each number condition (key) a start symbol rule
                        (value) to create sentences with incorrect verb number

    Returns:
        data_correct (list): tuples of (sentence, number_condition) for all
                            correct sentences
        data_incorrect (list): tuples of (sentence, number_condition) for all
                            sentences with number-verb disagreement
    """
    n_conditions = len(list(correct.keys())[0].split("_"))
    # Tasks that only have one noun of which we are tracking the number
    # Examples: simple, adv, qnty_simple, namepp
    if n_conditions == 1:
        grammar_correct, _ = get_grammar(grammar, correct["sg"])
    # Tasks that have two nouns of which we are tracking the number
    # Examples: nounpp
    elif n_conditions == 2:
        grammar_correct, _ = get_grammar(grammar, correct["sg_sg"])
    elif n_conditions == 3:
        grammar_correct, _ = get_grammar(grammar, correct["sg_sg_sg"])
    # Not tracking more than 3 nouns
    else:
        sys.exit("Number of conditions is incorrect. Please check the template.")

    correct_parsers = defaultdict()
    incorrect_parsers = defaultdict()
    data_correct, data_incorrect = [], []

    # 'corect' and 'incorrect' are dictionaries containing the same keys
    # Get the parsers for both the correct sentences and the incorrect
    # sentences, where the verb number does not match the noun number
    for corr_key, incorr_key in zip(correct, incorrect):
        _, correct_parsers[corr_key] = get_grammar(grammar, correct[corr_key])
        _, incorrect_parsers[incorr_key] = get_grammar(grammar, incorrect[incorr_key])

    # Generate n sentences and classify as either correct or incorrect
    for sent in tqdm(list(generate(grammar_correct, n=1000000))):
        for key in correct_parsers:
            # If a parser for correct sentence can parse the current sentence,
            # the sentence is correct
            if list(correct_parsers[key].parse(sent)):
                data_correct.append((" ".join(sent), key))
                break
            elif list(incorrect_parsers[key].parse(sent)):
                data_incorrect.append((" ".join(sent), key))
                break

    return data_correct, data_incorrect

def post_process(sentence):
    """
    Get index of the subject and its corresponding verb and clean up sentence.

    Args:
        sentence (str): uncapitalised string with * and ^ markers

    Returns:
        incomplete (str): sentence with first letter capitalised and ending
                        after the verb that is to be predicted
        subject_index (str): index of the subject
        verb_index (str): index of the corresponding verb
        complete (str): full sentence with capitalisation and ending with a
                        full stop (.)
    """
    sentence = sentence[0].upper() + sentence [1:]
    subject_index = sentence.split().index("*") - 1
    verb_index = sentence.split().index("^") - 2
    sentence = sentence.replace(" *", "").replace(" ^", "").split()

    complete = " ".join(sentence) + "."
    # Stop at the verb
    incomplete = " ".join(sentence[:verb_index+1])

    return incomplete, str(subject_index), str(verb_index), complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template", type=str, required=True,
                        help="The template of the output sentences.")
    parser.add_argument("-o", "--output", type=str, default="full_data",
                        help="Directory to store the full generated data.")
    parser.add_argument("--sample", default=False)
    parser.add_argument("--adverbs1_num", type=int, default=-1,
                        help="Maximum number of adverbs to use.")
    parser.add_argument("--position_nouns_num", type=int, default=-1,
                        help="Maximum number of position nouns to use.")
    parser.add_argument("--prepositions_num", type=int, default=-1,
                        help="Maximum number of prepositions to use.")
    parser.add_argument("--proper_nouns_num", type=int, default=-1,
                        help="Maximum number of proper nouns to use.")
    parser.add_argument("--subject_nouns_num", type=int, default=-1,
                        help="Maximum number of subject nouns to use.")
    parser.add_argument("--verbs_num", type=int, default=-1,
                        help="Maximum number of verbs to use.")
    parser.add_argument("--qnty_nouns_num", type=int, default=-1,
                        help="Maximum number of nouns to use for quantity pairs.")
    parser.add_argument("--object_nouns_num", type=int, default=-1,
                        help="Maximum number of object nouns to use.")
    args = parser.parse_args()

    # Read the vocabulary from the csv files
    adverbs = read_words("vocabulary/adverbs1.csv", args.adverbs1_num)
    conjunctions = read_words("vocabulary/conjunctions.csv")
    object_nouns = read_words("vocabulary/object_nouns.csv",
        args.object_nouns_num)
    position_nouns = read_words("vocabulary/position_nouns.csv",
        args.position_nouns_num)
    prepositions = read_words("vocabulary/prepositions.csv",
        args.prepositions_num)
    proper_nouns = read_words("vocabulary/proper_nouns.csv",
        args.proper_nouns_num)
    subject_nouns = read_words("vocabulary/subject_nouns.csv",
        args.subject_nouns_num)
    quantity_nouns = read_words("vocabulary/quantity_nouns.csv")
    quantity_subject_nouns = read_words("vocabulary/quantity_subject_nouns.csv",
        args.qnty_nouns_num)
    relative_pronouns = read_words("vocabulary/relative_pronouns.csv")
    verbs_trans = read_words("vocabulary/verbs_transitive.csv",
        args.verbs_num)
    verbs_intrans = read_words("vocabulary/verbs_intransitive.csv",
        args.verbs_num)
    verbs_modal = read_words("vocabulary/verbs_modal.csv")

    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{args.template}.tsv"

    abbreviations = {"sg":"singular", "pl": "plural"}

    print("Generating data and evaluating. This may take a while.")
    grammar, correct, incorrect = get_grammar_string(args.template, verbs_trans,
        verbs_intrans, subject_nouns, object_nouns, position_nouns, prepositions,
        adverbs, proper_nouns, quantity_nouns, quantity_subject_nouns,
        relative_pronouns, conjunctions, verbs_modal)

    data_correct, data_incorrect = generate_dataset(grammar, correct, incorrect)

    # Data comes in a tuple, with the second element indicating the condition,
    # e.g. sg_sg, which gives n_num = 2
    n_num = len(data_correct[0][1].split("_"))
    n_num = "".join([f"\tnumber{i}" for i in range(1,n_num+1)])
    header = f"agreement\tdisagreement\tcorrect_verb\tincorrect_verb\t"\
             f"subject_index\tverb_index{n_num}\tcompleted\n"

    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(header)
        for (agr, num1), (disagr, num2) in zip(data_correct, data_incorrect):
            assert num1 == num2
            # Get both correct and incorrect version of the same sentence
            agr, subject_idx, verb_idx, compl = post_process(agr)
            disagr, _, _, _ = post_process(disagr)
            corr_verb = agr.split()[int(verb_idx)]
            incorr_verb = disagr.split()[int(verb_idx)]
            # Turn "sg_sg" into ["singular", "singular"]
            numbers = [abbreviations[k] for k in num1.split("_")]
            line = [agr, disagr, corr_verb, incorr_verb, subject_idx, verb_idx]\
                   + numbers + [compl]
            f.write("\t".join(line) + "\n")
