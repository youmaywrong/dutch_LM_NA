"""
This module generates artificial natural language data, used to test for
the accuracy of subject-verb number agreement in sentences.
Currently, it implements functionality for 6 types of templates:
- simple: "the woman admires"
- adv: "the woman probably admires"
- adv_adv: "the woman now probably admires"
- adv_conjunction: "the woman probably and certainly admires"
- name_pp: "the woman near Mary admires"
- noun_pp: "the woman behind the cat admires"
- noun_pp_adv: "the woman behind the cat certainly admires"
Example uage:
- python generate.py -o generated_data -t name_pp
- python generate.py -o generated_data -t simple
etc.
"""

from collections import defaultdict

import os
import csv
import random
import argparse
import nltk

from nltk.parse.generate import generate
from tqdm import tqdm


def read_words(filename, n=-1):
    """Read vocabulary terms from a tsv file.
    The tsv's columns contain the different languages and / or number.
    Example 1: en\tnl
    Example 2: en_singular\ten_plural\tnl_singular\tplural
    Args:
        filename (str): tsv filename
        n (int): number of rows to sample to avoid combinatorial explosion
    Returns:
        dict: structure of { column_name : string formatted for nltk grammars }
    """
    columns = defaultdict(list)
    reader = list(csv.DictReader(open(filename), delimiter="\t"))
    n = len(reader) if n == -1 else min(n, len(reader))
    reader = random.sample(reader, n)
    for row in reader:
        for (k, v) in row.items():
            columns[k].append(f"'{v}'")
    return {k: " | ".join(v) for k, v in columns.items()}


def get_grammar(start, grammar):
    """Generate a grammar and parser given a starting symbol rule.
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


def generate_dataset(grammar, sg_correct, pl_correct, sg_incorrect,
                     pl_incorrect):
    """Generate data with correct and incorrect number-verb agreement.

    Args:
        sg_correct (str): start symbol rule for singular agreement
        pl_correct (str): start symbol rule for plural agreement
        sg_incorrect (str): start symbol rule for singular disagreement
        pl_incorrect (str): start symbol rule for plural disagreement
    Returns:
        data_correct (list of str): generated sentences with agreement
        data_incorrect (list of str): generated sentences with disagreement
    """
    # we need one grammar for generation and all parsers for classification
    grammar_sg_correct, parser_sg_correct = get_grammar(grammar, sg_correct)
    _, parser_pl_correct = get_grammar(grammar, pl_correct)
    _, parser_sg_incorrect = get_grammar(grammar, sg_incorrect)
    _, parser_pl_incorrect = get_grammar(grammar, pl_incorrect)

    # generate all possible sentences without looking at agreements
    data_correct, data_incorrect, data = [], [], []
    for sent in generate(grammar_sg_correct, n=1000000):
        data.append(sent)

    # check the feature grammars to categorise the sentence's (dis)agreement
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


def get_grammar_string(language, template, verbs, subject_nouns, object_nouns,
        position_nouns, prepositions, adverbs1, adverbs2, proper_nouns,
        conjunctions):
    """
    Generate a Feature-Based Grammar and output valid starting symbol rules
    for that grammar, for all languages in this codebase.
    Args:
        language (str): en | nl
        template (str): general name of syntactical construction e.g. "noun_pp"
        verbs (dict): language as key, list of words as values
        subject_nouns (dict): language as key, list of words as values
        position_nouns (dict): language as key, list of words as values
        prepositions (dict): language as key, list of words as values
        adverbs1 (dict): language as key, list of words as values
        adverbs2 (dict): language as key, list of words as values
        proper_nouns (dict): language as key, list of words as values
    Returns:
        grammar (str): overall grammar for the template
        sg_correct (str): starting rule for singular agreement
        pl_correct (str): starting rule for plural agreement
        sg_incorrect (str): starting rule for singular disagreement
        pl_incorrect (str): starting rule for plural disagreement
    """


    grammar = f"""
        VP[AGR=?a] -> V[AGR=?a]
        NP[AGR='sg'] -> {subject_nouns[f'{language}_singular']}
        NP[AGR='pl'] -> {subject_nouns[f'{language}_plural']}
        NP_obj[AGR='sg'] -> {object_nouns[f'{language}_singular']}
        V[AGR='pl'] -> {verbs[f'{language}_plural']}
        V[AGR='sg'] -> {verbs[f'{language}_singular']}
        PP -> P NP_pos
        PP_pn -> P PN
        PN -> {proper_nouns[language]}
        CONJ -> {conjunctions[language]}
        ADV1 -> {adverbs1[language]}
        ADV2 -> {adverbs2[language]}
        P -> {prepositions[language]}
        NP_pos -> {position_nouns[f'{language}_singular']}
        """
    print(grammar)
    adv = "ADV1"
    compl = "NP_obj[AGR='sg']"

    # SIMPLE template: "the woman admires"
    if template == "simple":
        sg_correct = f"S -> NP[AGR='sg']'*' VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' VP[AGR='sg']'^' {compl}"

    # ADV template: "the woman probably admires"
    elif template == "adv":
        sg_correct = f"S -> NP[AGR='sg']'*' '['{adv}']' VP[AGR='sg']'^' {compl} '='"
        pl_correct = f"S -> NP[AGR='pl']'*' '['{adv}']' VP[AGR='pl']'^' {compl} '='"
        sg_incorrect = f"S -> NP[AGR='sg']'*' '['{adv}']' VP[AGR='pl']'^' {compl} '='"
        pl_incorrect = f"S -> NP[AGR='pl']'*' '['{adv}']' VP[AGR='sg']'^' {compl} '='"

    # ADVADV template: "the woman now probably admires"
    elif template == "adv_adv":
        adv = "ADV2 ADV1"
        sg_correct = f"S -> NP[AGR='sg']'*' '['{adv}']' VP[AGR='sg']'^' {compl} '='"
        pl_correct = f"S -> NP[AGR='pl']'*' '['{adv}']' VP[AGR='pl']'^' {compl} '='"
        sg_incorrect = f"S -> NP[AGR='sg']'*' '['{adv}']' VP[AGR='pl']'^' {compl} '='"
        pl_incorrect = f"S -> NP[AGR='pl']'*' '['{adv}']' VP[AGR='sg']'^' {compl} '='"

    # ADVCONJUNCTION template: "the woman probably and certainly admires"
    elif template == "adv_conjunction":
        n = int(len(adverbs1[language].split(" | ")) / 2)
        grammar += f"""
            ADV3 -> {" | ".join(adverbs1[language].split(" | ")[:n])}
            ADV4 -> {" | ".join(adverbs1[language].split(" | ")[n:])}
        """
        adv = "ADV3 CONJ ADV4"
        sg_correct = f"S -> NP[AGR='sg']'*' '['{adv}']' VP[AGR='sg']'^' {compl} '='"
        pl_correct = f"S -> NP[AGR='pl']'*' '['{adv}']' VP[AGR='pl']'^' {compl} '='"
        sg_incorrect = f"S -> NP[AGR='sg']'*' '['{adv}']' VP[AGR='pl']'^' {compl} '='"
        pl_incorrect = f"S -> NP[AGR='pl']'*' '['{adv}']' VP[AGR='sg']'^' {compl} '='"

    # NAMEPP template: "the woman near Mary admires"
    elif template == "name_pp":
        sg_correct = f"S -> NP[AGR='sg']'*' PP_pn VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' PP_pn VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' PP_pn VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' PP_pn VP[AGR='sg']'^' {compl}"

    # NOUNPP template: "the woman behind the cat admires"
    elif template == "noun_pp":
        sg_correct = f"S -> NP[AGR='sg']'*' PP VP[AGR='sg']'^' {compl}"
        pl_correct = f"S -> NP[AGR='pl']'*' PP VP[AGR='pl']'^' {compl}"
        sg_incorrect = f"S -> NP[AGR='sg']'*' PP VP[AGR='pl']'^' {compl}"
        pl_incorrect = f"S -> NP[AGR='pl']'*' PP VP[AGR='sg']'^' {compl}"

    # NOUNPPADV template: "the woman behind the cat certainly admires"
    elif template == "noun_pp_adv":
        sg_correct = f"S -> NP[AGR='sg']'*' PP '['{adv}']' VP[AGR='sg']'^' {compl} '='"
        pl_correct = f"S -> NP[AGR='pl']'*' PP '['{adv}']' VP[AGR='pl']'^' {compl} '='"
        sg_incorrect = f"S -> NP[AGR='sg']'*' PP '['{adv}']' VP[AGR='pl']'^' {compl} '='"
        pl_incorrect = f"S -> NP[AGR='pl']'*' PP '['{adv}']' VP[AGR='sg']'^' {compl} '='"
    return grammar, sg_correct, pl_correct, sg_incorrect, pl_incorrect


def post_process(sentence, language):
    """
    Cleanup the sentence and extract the position of the subject and verb.
    Args:
        sentence (str): uncapitalised string with * and ^ markers
    Returns:
        sentence (str): clean sentence with captilisation
        subject_index (int): index of the subject
        verb_index (int): index of the verb
    """
    sentence = sentence[0].upper() + sentence[1:]

    # Move adverb in Dutch
    if language == "nl":
        adv = sentence[sentence.find("["):sentence.find("]") + 1]
        sentence = sentence.replace(adv, "").replace("=", adv)
    sentence = sentence.replace('[', "").replace("]", "").replace("=", "")

    # Find subject and verb indices
    subject_index = sentence.split().index("*") - 1
    verb_index = sentence.split().index("^") - 2
    sentence = sentence.replace(" *", "").replace(" ^", "").split()

    # Create incomplete and complete sentences
    completed = " ".join(sentence) + "."
    sentence = " ".join(sentence[:verb_index+1])
    return sentence, str(subject_index), str(verb_index), completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template", type=str, required=True,
                        choices=["simple", "noun_pp", "noun_pp_adv", "name_pp",
                                 "adv_adv", "adv", "adv_conjunction"],
                        help="The template of the output sentences.")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Folder name to store the generated data in.")
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
    args = parser.parse_args()

    # retrieve the vocabulary to use form tsv files
    adverbs1 = read_words("vocabulary/adverbs1.tsv", args.adverbs1_num)
    adverbs2 = read_words("vocabulary/adverbs2.tsv")
    conjunctions = read_words("vocabulary/conjunctions.tsv")
    object_nouns = read_words("vocabulary/object_nouns.tsv")
    position_nouns = read_words(
        "vocabulary/position_nouns.tsv", args.position_nouns_num)
    prepositions = read_words("vocabulary/prepositions.tsv",
        args.prepositions_num)
    proper_nouns = read_words("vocabulary/proper_nouns.tsv",
        args.proper_nouns_num)
    subject_nouns = read_words(
        "vocabulary/subject_nouns.tsv", args.subject_nouns_num)
    verbs = read_words("vocabulary/verbs.tsv", args.verbs_num)

    # iterate over languages, currently NL and EN implemented
    for language in ["nl", "en"]:
        print(f"Generating data in {language}. This may take a while.")

        # generate the grammar to use
        grammar, sg_correct, pl_correct, sg_incorrect, pl_incorrect = \
            get_grammar_string(language, args.template, verbs, subject_nouns,
                               object_nouns, position_nouns, prepositions,
                               adverbs1, adverbs2, proper_nouns, conjunctions)

        # generate the data from the grammar
        data_correct, data_incorrect = generate_dataset(
            grammar, sg_correct=sg_correct, pl_correct=pl_correct,
            sg_incorrect=sg_incorrect, pl_incorrect=pl_incorrect
        )

        # postprocess and save in file
        data = []
        for (agr, num1), (disagr, num2) in zip(data_correct, data_incorrect):
            assert num1 == num2
            agr, subject_index, verb_index, compl = post_process(agr, language)
            disagr, _, _, _ = post_process(disagr, language)
            data.append((agr, disagr, str(num1), subject_index, verb_index, compl))

        # filename = f"{language}_{args.template}.tsv"
        # header = "agreement\tdisagreement\tnumber\tsubject_index" + \
        #     "\tverb_index\tcompleted\n"
        # with open(os.path.join(args.output, filename), 'w') as f:
        #     f.write(header)
        #     for line in data:
        #         f.write('\t'.join(line) + "\n")
