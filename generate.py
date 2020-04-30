import os, sys
import csv
import random
import argparse
import nltk

from collections import defaultdict
from nltk.parse.generate import generate
from tqdm import tqdm

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


def get_grammar(start, grammar):
    """
    Generate a grammar and parser givven a starting symbol rule.
    Args:
        start (str): grammar rule such as S -> N[AGR=?a]
    Returns:
        g (nltk.grammar.FeatureGrammar): grammar in string, one rule per line
        p (nltk.parse.FeatureEarleyChartParser): parser created from grammar
    """
    g = f"% start S\n{start}\n{grammar}"
    g = nltk.grammar.FeatureGrammar.fromstring(g)
    p = nltk.parse.FeatureEarleyChartParser(g)

    return g, p


def get_grammar_string(template, verbs, subject_nouns, object_nouns,
        position_nouns, prepositions, adverbs1, adverbs2, proper_nouns,
        quantity_nouns, quantity_subject_nouns, relative_pronouns, conjunctions):
    """
    Generate a Feature-Based Grammar and output valid starting symbol rules for
    that grammar.
    Args:
        template (str): general name of syntactical construction, e.g. "noun_pp"
    """

    grammar = f"""
        VP[AGR=?a] -> V[AGR=?a]
        VP_conj[AGR=?a] -> V[AGR=?a] CONJ V[AGR=?a]
        REL[AGR=?a] -> REL_pn NP_obj[AGR=?a] NP_obj VP[AGR=?a]
        REL_adv[AGR=?a] -> REL_pn NP_obj[AGR=?a] NP_obj ADV1 VP[AGR=?a]
        NP[AGR='sg'] -> {subject_nouns["singular"]}
        NP[AGR='pl'] -> {subject_nouns["plural"]}
        NP_obj[AGR='sg'] -> {object_nouns["singular"]}
        NP_obj[AGR='pl'] -> {object_nouns["plural"]}
        V[AGR='sg'] -> {verbs["singular"]}
        V[AGR='pl'] -> {verbs["plural"]}
        PP -> P NP_pos
        PP_pn -> P PN
        PN -> {proper_nouns["proper_noun"]}
        CONJ -> {conjunctions["conjunction"]}
        ADV1 -> {adverbs1["adverb"]}
        ADV2 -> {adverbs2["adverb"]}
        P -> {prepositions["preposition"]}
        NP_pos -> {position_nouns["singular"]}
        QNTY[AGR='sg'] -> {quantity_nouns["singular"]}
        QNTY[AGR='pl'] -> {quantity_nouns["plural"]}
        QNTY_subj -> {quantity_subject_nouns["quantity_subject_noun"]}
        REL_pn -> 'van wie' | 'die'
    """
    # For some reason, the plural position nouns have been ommitted.
    adv = "ADV1"
    compl = "NP_obj[AGR='sg']"

    if template == "simple":
        sg_correct = f"S -> NP[AGR='sg']'*' VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' VP[AGR='sg']'^' {compl}"

    elif template == "adv":
        sg_correct = f"S -> NP[AGR='sg']'*' VP[AGR='sg']'^' {compl} {adv}"
        pl_correct = f"S -> NP[AGR='pl']'*' VP[AGR='pl']'^' {compl} {adv}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' VP[AGR='pl']'^' {compl} {adv}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' VP[AGR='sg']'^' {compl} {adv}"

    elif template == "adv_adv":
        adv = "ADV2 ADV1"
        sg_correct = f"S -> NP[AGR='sg']'*' VP[AGR='sg']'^' {compl} {adv}"
        pl_correct = f"S -> NP[AGR='pl']'*' VP[AGR='pl']'^' {compl} {adv}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' VP[AGR='pl']'^' {compl} {adv}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' VP[AGR='sg']'^' {compl} {adv}"

    elif template == "adv_conjunction":
        adv = "ADV1 CONJ ADV1"
        sg_correct = f"S -> NP[AGR='sg']'*' VP[AGR='sg']'^' {compl} {adv}"
        pl_correct = f"S -> NP[AGR='pl']'*' VP[AGR='pl']'^' {compl} {adv}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' VP[AGR='pl']'^' {compl} {adv}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' VP[AGR='sg']'^' {compl} {adv}"

    elif template == "name_pp":
        sg_correct = f"S -> NP[AGR='sg']'*' PP_pn VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' PP_pn VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' PP_pn VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' PP_pn VP[AGR='sg']'^' {compl}"

    elif template == "noun_pp":
        sg_correct = f"S -> NP[AGR='sg']'*' PP VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' PP VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' PP VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' PP VP[AGR='sg']'^' {compl}"

    elif template == "noun_pp_adv":
        sg_correct = f"S -> NP[AGR='sg']'*' PP VP[AGR='sg']'^' {compl} {adv}"
        pl_correct = f"S -> NP[AGR='pl']'*' PP VP[AGR='pl']'^' {compl} {adv}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' PP VP[AGR='pl']'^' {compl} {adv}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' PP VP[AGR='sg']'^' {compl} {adv}"

    elif template == "qnty_simple":
        sg_correct = f"S -> QNTY[AGR='sg']'*' QNTY_subj VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> QNTY[AGR='pl']'*' QNTY_subj VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> QNTY[AGR='sg']'*' QNTY_subj VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> QNTY[AGR='pl']'*' QNTY_subj VP[AGR='sg']'^' {compl}"

    elif template == "qnty_noun_pp":
        sg_correct = f"S -> QNTY[AGR='sg']'*' QNTY_subj PP VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> QNTY[AGR='pl']'*' QNTY_subj PP VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> QNTY[AGR='sg']'*' QNTY_subj PP VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> QNTY[AGR='pl']'*' QNTY_subj PP VP[AGR='sg']'^' {compl}"

    elif template == "qnty_name_pp":
        sg_correct = f"S -> QNTY[AGR='sg']'*' QNTY_subj PP_pn VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> QNTY[AGR='pl']'*' QNTY_subj PP_pn VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> QNTY[AGR='sg']'*' QNTY_subj PP_pn VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> QNTY[AGR='pl']'*' QNTY_subj PP_pn VP[AGR='sg']'^' {compl}"

    elif template == "verb_conj":
        sg_correct = f"S -> NP[AGR='sg']'*' VP_conj[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' VP_conj[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' VP_conj[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' VP_conj[AGR='sg']'^' {compl}"

    elif template == "rel_sg":
        sg_correct = f"S -> NP[AGR='sg']'*' REL[AGR='sg'] VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' REL[AGR='sg'] VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' REL[AGR='sg'] VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' REL[AGR='sg'] VP[AGR='sg']'^' {compl}"

    elif template == "rel_sg_adv":
        sg_correct = f"S -> NP[AGR='sg']'*' REL_adv[AGR='sg'] VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' REL_adv[AGR='sg'] VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' REL_adv[AGR='sg'] VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' REL_adv[AGR='sg'] VP[AGR='sg']'^' {compl}"

    elif template == "rel_pl":
        sg_correct = f"S -> NP[AGR='sg']'*' REL[AGR='pl'] VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' REL[AGR='pl'] VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' REL[AGR='pl'] VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' REL[AGR='pl'] VP[AGR='sg']'^' {compl}"

    elif template == "rel_pl_adv":
        sg_correct = f"S -> NP[AGR='sg']'*' REL_adv[AGR='pl'] VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' REL_adv[AGR='pl'] VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' REL_adv[AGR='pl'] VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' REL_adv[AGR='pl'] VP[AGR='sg']'^' {compl}"

    else:
        sys.exit("No valid template")

    return grammar, sg_correct, pl_correct, sg_incorrect, pl_incorrect


def generate_dataset(grammar, sg_correct, pl_correct, sg_incorrect,
        pl_incorrect):
    """
    """
    # We need one grammar for generation and all parsers for classification
    # grammar_sg_correct, parser_sg_correct = get_grammar(grammar, sg_correct)
    grammar_sg_correct, parser_sg_correct = get_grammar(grammar, sg_correct)
    _, parser_pl_correct = get_grammar(grammar, pl_correct)
    _, parser_sg_incorrect = get_grammar(grammar, sg_incorrect)
    _, parser_pl_incorrect = get_grammar(grammar, pl_incorrect)

    data_correct, data_incorrect = [], []
    for sent in tqdm(list(generate(grammar_sg_correct, n=1000000))):
        if list(parser_sg_correct.parse(sent)):
            data_correct.append((" ".join(sent), "singular"))
        elif list(parser_pl_correct.parse(sent)):
            data_correct.append((" ".join(sent), "plural"))
        elif list(parser_sg_incorrect.parse(sent)):
            data_incorrect.append((" ".join(sent), "singular"))
        elif list(parser_pl_incorrect.parse(sent)):
            data_incorrect.append((" ".join(sent), "plural"))

    return data_correct, data_incorrect


def post_process(sentence):
    """
    """
    if sentence[0] == " ":
        sentence = sentence[2:]
    sentence = sentence[0].upper() + sentence [1:]
    subject_index = sentence.split().index("*") - 1
    verb_index = sentence.split().index("^") - 2
    sentence = sentence.replace(" *", "").replace(" ^", "").split()

    complete = " ".join(sentence) + "."
    incomplete = " ".join(sentence[:verb_index+1])

    return incomplete, str(subject_index), str(verb_index), complete

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template", type=str, required=True,
                        choices=["simple", "noun_pp", "noun_pp_adv", "name_pp",
                                 "adv_adv", "adv", "adv_conjunction",
                                 "qnty_simple", "rel_sg", "rel_sg_adv",
                                 "rel_pl", "rel_pl_adv", "verb_conj",
                                 "qnty_noun_pp", "qnty_name_pp"],
                        help="The template of the output sentences.")
    parser.add_argument("-o", "--output", type=str, default="generated_data",
                        help="Directory to store the generated data.")
    parser.add_argument("--adverbs1_num", type=int, default=-1,
                        help="Maximum number of adverbs1 to use.")
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
    args = parser.parse_args()

    # Read the vocabulary from the csv files
    adverbs1 = read_words("vocabulary/adverbs1.csv", args.adverbs1_num)
    adverbs2 = read_words("vocabulary/adverbs2.csv")
    conjunctions = read_words("vocabulary/conjunctions.csv")
    object_nouns = read_words("vocabulary/object_nouns.csv")
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
    verbs = read_words("vocabulary/verbs.csv", args.verbs_num)

    print("Generating data and evaluating. This may take a while.")
    grammar, sg_correct, pl_correct, sg_incorrect, pl_incorrect = \
        get_grammar_string(args.template, verbs, subject_nouns,
                           object_nouns, position_nouns, prepositions,
                           adverbs1, adverbs2, proper_nouns, quantity_nouns,
                           quantity_subject_nouns, relative_pronouns,
                           conjunctions)

    data_correct, data_incorrect = generate_dataset(grammar,
        sg_correct=sg_correct, pl_correct=pl_correct, sg_incorrect=sg_incorrect,
        pl_incorrect=pl_incorrect)

    data = []
    for (agr, num1), (disagr, num2) in zip(data_correct, data_incorrect):
        assert num1 == num2
        agr, subject_index, verb_index, compl = post_process(agr)
        disagr, _, _, _ = post_process(disagr)
        data.append((agr, disagr, str(num1), subject_index, verb_index, compl))

    output_dir = args.output
    filename = f"{args.template}.csv"
    header = "agreement,disagreement,number,subject_index,verb_index,completed\n"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(header)
        for line in data:
            f.write(",".join(line) + "\n")
