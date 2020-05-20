from collections import defaultdict
import nltk
import sys

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


def get_opposite_number(grammatical_number):
    """
    Args:
        grammatical_number (str): "pl" or "sg"
    Returns:
        pl (str) if grammatical_number is sg
        sg (str) if grammatical number if pl
    """
    if grammatical_number not in ["sg", "pl"]:
        sys.exit("Grammatical number is not singular or plural.")
    return "sgpl".replace(grammatical_number, "")


def get_grammar_string(template, verbs_trans, verbs_intrans, subject_nouns,
        object_nouns, position_nouns, prepositions, adverbs, proper_nouns,
        quantity_nouns, quantity_subject_nouns, relative_pronouns, conjunctions,
        verbs_modal):
    """
    Generate a Feature-Based Grammar and output valid starting symbol rules for
    that grammar.
    Args:
        template (str): general name of syntactical construction, e.g. "nounpp"

    Returns:
        grammar (str): NTLK feature grammar
        correct (dict): for each number condition (key) a start symbol rule
                        (value) to create sentences with noun-verb agreement
        incorrect (dict): for each number condition (key) a start symbol rule
                        (value) to create sentences with incorrect verb number
    """
    correct, incorrect = defaultdict(str), defaultdict(str)
    conditions = ["sg", "pl"]

    grammar = f"""
        VP[AGR=?a] -> V[AGR=?a]
        VP_intrans[AGR=?a] -> V_intrans[AGR=?a]
        VP_conj[AGR=?a] -> V[AGR=?a] CONJ V[AGR=?a]
        REL[AGR=?a] -> REL_pn NP_obj[AGR=?a] NP_obj VP[AGR=?a]
        REL_adv[AGR=?a] -> REL_pn NP_obj[AGR=?a] NP_obj ADV VP[AGR=?a]
        NP[AGR='sg'] -> {subject_nouns["singular"]}
        NP[AGR='pl'] -> {subject_nouns["plural"]}
        NP_obj[AGR='sg'] -> {object_nouns["singular"]}
        NP_obj[AGR='pl'] -> {object_nouns["plural"]}
        V[AGR='sg'] -> {verbs_trans["singular"]}
        V[AGR='pl'] -> {verbs_trans["plural"]}
        V_intrans[AGR='sg'] -> {verbs_intrans["singular"]}
        V_intrans[AGR='pl'] -> {verbs_intrans["plural"]}
        V_that[AGR='sg'] -> {verbs_modal["singular"]}
        V_that[AGR='pl'] -> {verbs_modal["plural"]}
        PP[AGR=?a] -> P NP_pos[AGR=?a]
        PP_pn -> P PN
        PN -> {proper_nouns["proper_noun"]}
        CONJ -> {conjunctions["conjunction"]}
        ADV -> {adverbs["adverb"]}
        P -> {prepositions["preposition"]}
        NP_pos[AGR='sg'] -> {position_nouns["singular"]}
        NP_pos[AGR='pl'] -> {position_nouns["plural"]}
        QNTY[AGR='sg'] -> {quantity_nouns["singular"]}
        QNTY[AGR='pl'] -> {quantity_nouns["plural"]}
        QNTY_subj -> {quantity_subject_nouns["noun"]}
        REL_pn -> 'van wie' | 'die'
        COMPL -> 'de persoon'
    """

    if template == "simple":
        for num in conditions:
            wrong_num = get_opposite_number(num)
            correct[f"{num}"] = f"S -> NP[AGR={num}]'*' "\
                    f"VP[AGR={num}]'^' COMPL"
            incorrect[f"{num}"] = f"S -> NP[AGR={num}]'*' "\
                    f"VP[AGR={wrong_num}]'^' COMPL"

    elif template == "adv":
        for num in conditions:
            wrong_num = get_opposite_number(num)
            correct[f"{num}"] = f"S -> NP[AGR={num}]'*' "\
                    f"VP[AGR={num}]'^' COMPL ADV"
            incorrect[f"{num}"] = f"S -> NP[AGR={num}]'*' "\
                    f"VP[AGR={wrong_num}]'^' COMPL ADV"

    elif template == "namepp":
        for num in conditions:
            wrong_num = get_opposite_number(num)
            correct[f"{num}"] = f"S -> NP[AGR={num}]'*' PP_pn "\
                    f"VP[AGR={num}]'^' COMPL"
            incorrect[f"{num}"] = f"S -> NP[AGR={num}]'*' PP_pn "\
                    f"VP[AGR={wrong_num}]'^' COMPL"

    elif template == "qnty_simple":
        for num in conditions:
            wrong_num = get_opposite_number(num)
            correct[f"{num}"] = f"S -> QNTY[AGR={num}]'*' QNTY_subj"\
                    f" VP[AGR={num}]'^' COMPL"
            incorrect[f"{num}"] = f"S -> QNTY[AGR={num}]'*' QNTY_subj"\
                    f" VP[AGR={wrong_num}]'^' COMPL"

    elif template == "qnty_namepp":
        for num in conditions:
            wrong_num = get_opposite_number(num)
            correct[f"{num}"] = f"S -> QNTY[AGR={num}]'*' QNTY_subj"\
                    f" PP_pn VP[AGR={num}]'^' COMPL"
            incorrect[f"{num}"] = f"S -> QNTY[AGR={num}]'*' QNTY_subj"\
                    f" PP_pn VP[AGR={wrong_num}]'^' COMPL"

    elif template == "nounpp":
        for num in conditions:
            wrong_num = get_opposite_number(num)
            for pp_num in conditions:
                correct[f"{num}_{pp_num}"] = f"S -> NP[AGR={num}]"\
                        f"'*' PP[AGR={pp_num}] VP[AGR={num}]'^' COMPL"
                incorrect[f"{num}_{pp_num}"] = f"S -> NP[AGR={num}]"\
                        f"'*' PP[AGR={pp_num}] VP[AGR={wrong_num}]'^' COMPL"

    elif template == "qnty_nounpp":
        for num in conditions:
            wrong_num = get_opposite_number(num)
            for pp_num in conditions:
                correct[f"{num}_{pp_num}"] = f"S -> QNTY[AGR={num}]"\
                        f"'*' QNTY_subj PP[AGR={pp_num}] VP[AGR={num}]'^'"\
                        f" COMPL"
                incorrect[f"{num}_{pp_num}"] = f"S -> QNTY[AGR={num}]"\
                        f"'*' QNTY_subj PP[AGR={pp_num}] VP[AGR={wrong_num}]'^'"\
                        f" COMPL"

    elif template == "that_trans":
        for num1 in conditions:
            for num2 in conditions:
                wrong_num = get_opposite_number(num2)
                correct[f"{num1}_{num2}"] = f"S -> NP_obj[AGR={num1}]"\
                        f"V_that[AGR={num1}] NP[AGR={num2}]'*' COMPL "\
                        f"VP[AGR={num2}]'^' "
                incorrect[f"{num1}_{num2}"] = f"S -> NP_obj[AGR={num1}]"\
                        f"V_that[AGR={num1}] NP[AGR={num2}]'*' COMPL "\
                        f"VP[AGR={wrong_num}]'^'"

    elif template == "that_simple":
        for num1 in conditions:
            for num2 in conditions:
                wrong_num = get_opposite_number(num2)
                correct[f"{num1}_{num2}"] = f"S -> NP_obj[AGR={num1}]"\
                        f"V_that[AGR={num1}] NP[AGR={num2}]'*' "\
                        f"VP_intrans[AGR={num2}]'^' "
                incorrect[f"{num1}_{num2}"] = f"S -> NP_obj[AGR={num1}]"\
                        f"V_that[AGR={num1}] NP[AGR={num2}]'*' "\
                        f"VP_intrans[AGR={wrong_num}]'^'"


    elif template == "that_adv":
        for num1 in conditions:
            for num2 in conditions:
                wrong_num = get_opposite_number(num2)
                correct[f"{num1}_{num2}"] = f"S -> NP_obj[AGR={num1}]"\
                        f"V_that[AGR={num1}] NP[AGR={num2}]'*' ADV "\
                        f"VP_intrans[AGR={num2}]'^' "
                incorrect[f"{num1}_{num2}"] = f"S -> NP_obj[AGR={num1}]"\
                        f"V_that[AGR={num1}] NP[AGR={num2}]'*' ADV "\
                        f"VP_intrans[AGR={wrong_num}]'^'"

    elif template == "that_nounpp":
        for num1 in conditions:
            for num2 in conditions:
                wrong_num = get_opposite_number(num2)
                for num3 in conditions:
                    correct[f"{num1}_{num2}_{num3}"] =\
                            f"S -> NP_obj[AGR={num1}] "\
                            f"V_that[AGR={num1}] NP[AGR={num2}]'*' "\
                            f"PP[AGR={num3}] VP_intrans[AGR={num2}]'^' "
                    incorrect[f"{num1}_{num2}_{num3}"] =\
                            f"S -> NP_obj[AGR={num1}] "\
                            f"V_that[AGR={num1}] NP[AGR={num2}]'*' "\
                            f"PP[AGR={num3}] VP_intrans[AGR={wrong_num}]'^' "\

    elif template == "noun_conj":
        for num1 in conditions:
            for num2 in conditions:
                correct[f"{num1}_{num2}"] = f"S -> NP[AGR={num1}]"\
                        f" 'en' NP[AGR={num2}]'*' VP[AGR='pl']'^' COMPL"
                incorrect[f"{num1}_{num2}"] = f"S -> NP[AGR={num1}]"\
                        f" 'en' NP[AGR={num2}]'*' VP[AGR='sg']'^' COMPL"

    elif template == "s_conj":
        for num1 in conditions:
            for num2 in conditions:
                wrong_num = get_opposite_number(num2)
                correct[f"{num1}_{num2}"] = f"S -> NP[AGR={num1}]"\
                        f" VP[AGR={num1}] 'en' NP[AGR={num2}]'*' "\
                        f"VP[AGR={num2}]'^' COMPL"
                incorrect[f"{num1}_{num2}"] = f"S -> NP[AGR={num1}]"\
                        f" VP[AGR={num1}] 'en' NP[AGR={num2}]'*' "\
                        f"VP[AGR={wrong_num}]'^' COMPL"

    elif template == "rel_def":
        for num1 in conditions:
            wrong_num = get_opposite_number(num1)
            correct[f"{num1}"] = f"S -> NP[AGR={num1}]'*' 'die' "\
                    f"V_intrans[AGR={num1}] ',' VP[AGR={num1}]'^' COMPL"
            incorrect[f"{num1}"] = f"S -> NP[AGR={num1}]'*' 'die' "\
                    f"V_intrans[AGR={num1}] ',' VP[AGR={wrong_num}]'^' COMPL"\

    elif template == "rel_nondef":
        for num1 in conditions:
            wrong_num = get_opposite_number(num1)
            correct[f"{num1}"] = f"S -> NP[AGR={num1}]'*' ',' 'die' "\
                    f"V_intrans[AGR={num1}] ',' VP[AGR={num1}]'^' COMPL"
            incorrect[f"{num1}"] = f"S -> NP[AGR={num1}]'*' ',' 'die' "\
                    f"V_intrans[AGR={num1}] ',' VP[AGR={wrong_num}]'^' COMPL"\

    elif template == "rel_def_obj":
        for num1 in conditions:
            wrong_num = get_opposite_number(num1)
            for num2 in conditions:
                correct[f"{num1}_{num2}"] = f"S -> NP[AGR={num1}]'*' 'die' "\
                        f"NP_obj[AGR={num2}] VP[AGR={num2}] ',' VP[AGR={num1}]"\
                        f"'^' COMPL"
                incorrect[f"{num1}_{num2}"] = f"S -> NP[AGR={num1}]'*' 'die' "\
                        f"NP_obj[AGR={num2}] VP[AGR={num2}] ',' VP[AGR={wrong_num}]"\
                        f"'^' COMPL"

    else:
        sys.exit("No valid template")

    return grammar, correct, incorrect
