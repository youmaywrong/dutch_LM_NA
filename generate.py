import os
import csv
import random
import argparse
import nltk
import csv

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
        REL_poss[AGR=?a] -> REL_pn NP_obj[AGR=?a] VP[AGR=?a]
        NP[AGR='sg'] -> {subject_nouns["singular"]}
        NP[AGR='pl'] -> {subject_nouns["plural"]}
        NP_obj[AGR='sg'] -> {object_nouns["singular"]}
        NP_obj[AGR='pl'] -> {object_nouns["plural"]}
        V[AGR='sg'] -> {verbs["singular"]}
        V[AGR='pl'] -> {verbs["plural"]}
        PP -> NP_pos
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
        REL_pn -> 'van wie'
        DET[AGR='pl'] -> 'de'
        DET[AGR='sg'] -> ' '
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

    elif template == "rel_clause_sg":
        sg_correct = f"S -> NP[AGR='sg']'*' REL_poss[AGR='sg'] VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' REL_poss[AGR='sg'] VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' REL_poss[AGR='sg'] VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' REL_poss[AGR='sg'] VP[AGR='sg']'^' {compl}"

    elif template == "rel_clause_pl":
        sg_correct = f"S -> NP[AGR='sg']'*' REL_poss[AGR='pl'] VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' REL_poss[AGR='pl'] VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' REL_poss[AGR='pl'] VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' REL_poss[AGR='pl'] VP[AGR='sg']'^' {compl}"


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

    data, data_correct, data_incorrect = [], [], []
    for sentence in generate(grammar_sg_correct, n=10000):
        data.append(sentence)

    for sent in tqdm(data):
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
    adverbs1 = read_words("vocabulary/adverbs1.csv")
    adverbs2 = read_words("vocabulary/adverbs2.csv")
    conjunctions = read_words("vocabulary/conjunctions.csv")
    object_nouns = read_words("vocabulary/object_nouns.csv")
    position_nouns = read_words("vocabulary/position_nouns.csv")
    prepositions = read_words("vocabulary/prepositions.csv")
    proper_nouns = read_words("vocabulary/proper_nouns.csv")
    subject_nouns = read_words("vocabulary/subject_nouns.csv")
    quantity_nouns = read_words("vocabulary/quantity_nouns.csv")
    quantity_subject_nouns = read_words("vocabulary/quantity_subject_nouns.csv")
    relative_pronouns = read_words("vocabulary/relative_pronouns.csv")
    verbs = read_words("vocabulary/verbs.csv")

    template = "rel_clause_sg"

    print("Generating data. This may take a while.")
    grammar, sg_correct, pl_correct, sg_incorrect, pl_incorrect = \
        get_grammar_string(template, verbs, subject_nouns,
                           object_nouns, position_nouns, prepositions,
                           adverbs1, adverbs2, proper_nouns, quantity_nouns,
                           quantity_subject_nouns, relative_pronouns,
                           conjunctions)


    data_correct, data_incorrect = generate_dataset(
        grammar, sg_correct=sg_correct, pl_correct=pl_correct,
        sg_incorrect=sg_incorrect, pl_incorrect=pl_incorrect
    )

    data = []
    for (agr, num1), (disagr, num2) in zip(data_correct, data_incorrect):
        assert num1 == num2
        agr, subject_index, verb_index, compl = post_process(agr)
        disagr, _, _, _ = post_process(disagr)
        data.append((agr, disagr, str(num1), subject_index, verb_index, compl))

    for i in data:
        print(i)
