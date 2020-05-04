from collections import defaultdict
import nltk

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
        quantity_nouns, quantity_subject_nouns, relative_pronouns, conjunctions, verbs_modal):
    """
    Generate a Feature-Based Grammar and output valid starting symbol rules for
    that grammar.
    Args:
        template (str): general name of syntactical construction, e.g. "nounpp"
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
        for subj_num in conditions:
            wrong_num = get_opposite_number(subj_num)
            correct[f"{subj_num}"] = f"S -> NP[AGR={subj_num}]'*' "\
                    f"VP[AGR={subj_num}]'^' COMPL"
            incorrect[f"{subj_num}"] = f"S -> NP[AGR={subj_num}]'*' "\
                    f"VP[AGR={wrong_num}]'^' COMPL"

    elif template == "adv":
        for subj_num in conditions:
            wrong_num = get_opposite_number(subj_num)
            correct[f"{subj_num}"] = f"S -> NP[AGR={subj_num}]'*' "\
                    f"VP[AGR={subj_num}]'^' COMPL ADV"
            incorrect[f"{subj_num}"] = f"S -> NP[AGR={subj_num}]'*' "\
                    f"VP[AGR={wrong_num}]'^' COMPL ADV"

    elif template == "namepp":
        for subj_num in conditions:
            wrong_num = get_opposite_number(subj_num)
            correct[f"{subj_num}"] = f"S -> NP[AGR={subj_num}]'*' PP_pn "\
                    f"VP[AGR={subj_num}]'^' COMPL"
            incorrect[f"{subj_num}"] = f"S -> NP[AGR={subj_num}]'*' PP_pn "\
                    f"VP[AGR={wrong_num}]'^' COMPL"

    elif template == "qnty_simple":
        for subj_num in conditions:
            wrong_num = get_opposite_number(subj_num)
            correct[f"{subj_num}"] = f"S -> QNTY[AGR={subj_num}]'*' QNTY_subj"\
                    f" VP[AGR={subj_num}]'^' COMPL"
            incorrect[f"{subj_num}"] = f"S -> QNTY[AGR={subj_num}]'*' QNTY_subj"\
                    f" VP[AGR={wrong_num}]'^' COMPL"

    elif template == "qnty_namepp":
        for subj_num in conditions:
            wrong_num = get_opposite_number(subj_num)
            correct[f"{subj_num}"] = f"S -> QNTY[AGR={subj_num}]'*' QNTY_subj"\
                    f" PP_pn VP[AGR={subj_num}]'^' COMPL"
            incorrect[f"{subj_num}"] = f"S -> QNTY[AGR={subj_num}]'*' QNTY_subj"\
                    f" PP_pn VP[AGR={wrong_num}]'^' COMPL"

    elif template == "nounpp":
        for subj_num in conditions:
            wrong_num = get_opposite_number(subj_num)
            for pp_num in conditions:
                correct[f"{subj_num}_{pp_num}"] = f"S -> NP[AGR={subj_num}]"\
                        f"'*' PP[AGR={pp_num}] VP[AGR={subj_num}]'^' COMPL"
                incorrect[f"{subj_num}_{pp_num}"] = f"S -> NP[AGR={subj_num}]"\
                        f"'*' PP[AGR={pp_num}] VP[AGR={wrong_num}]'^' COMPL"

    elif template == "qnty_nounpp":
        for subj_num in conditions:
            wrong_num = get_opposite_number(subj_num)
            for pp_num in conditions:
                correct[f"{subj_num}_{pp_num}"] = f"S -> QNTY[AGR={subj_num}]"\
                        f"'*' QNTY_subj PP[AGR={pp_num}] VP[AGR={subj_num}]'^'"\
                        f" COMPL"
                incorrect[f"{subj_num}_{pp_num}"] = f"S -> QNTY[AGR={subj_num}]"\
                        f"'*' QNTY_subj PP[AGR={pp_num}] VP[AGR={wrong_num}]'^'"\
                        f" COMPL"

    elif template == "that":
        for sec_num in conditions:
            wrong_num = get_opposite_number(sec_num)
            for first_num in conditions:
                # Mention the second number first, as we are tracking the number
                # agreement for that noun
                correct[f"{sec_num}_{first_num}"] = f"S -> NP_obj[AGR={first_num}]"\
                        f"V_that[AGR={first_num}] NP[AGR={sec_num}]'*' "\
                        f"VP_intrans[AGR={sec_num}]'^' "
                incorrect[f"{sec_num}_{first_num}"] = f"S -> NP_obj[AGR={first_num}]"\
                        f"V_that[AGR={first_num}] NP[AGR={sec_num}]'*' "\
                        f"VP_intrans[AGR={wrong_num}]'^'"

    elif template == "that_compl":
        for sec_num in conditions:
            wrong_num = get_opposite_number(sec_num)
            for first_num in conditions:
                # Mention the second number first, as we are tracking the number
                # agreement for that noun
                correct[f"{sec_num}_{first_num}"] = f"S -> NP_obj[AGR={first_num}]"\
                        f"V_that[AGR={first_num}] NP[AGR={sec_num}]'*' COMPL "\
                        f"VP[AGR={sec_num}]'^' "
                incorrect[f"{sec_num}_{first_num}"] = f"S -> NP_obj[AGR={first_num}]"\
                        f"V_that[AGR={first_num}] NP[AGR={sec_num}]'*' COMPL "\
                        f"VP[AGR={wrong_num}]'^'"

    elif template == "that_adv":
        for sec_num in conditions:
            wrong_num = get_opposite_number(sec_num)
            for first_num in conditions:
                # Mention the second number first, as we are tracking the number
                # agreement for that noun
                correct[f"{sec_num}_{first_num}"] = f"S -> NP_obj[AGR={first_num}]"\
                        f"V_that[AGR={first_num}] NP[AGR={sec_num}]'*' ADV COMPL "\
                        f"VP[AGR={sec_num}]'^' "
                incorrect[f"{sec_num}_{first_num}"] = f"S -> NP_obj[AGR={first_num}]"\
                        f"V_that[AGR={first_num}] NP[AGR={sec_num}]'*' ADV COMPL "\
                        f"VP[AGR={wrong_num}]'^'"

    elif template == "that_nounpp":
        for sec_num in conditions:
            wrong_num = get_opposite_number(sec_num)
            for first_num in conditions:
                for third_num in conditions:
                    # Mention the second number first, as we are tracking the number
                    # agreement for that noun
                    correct[f"{sec_num}_{first_num}_{third_num}"] =\
                            f"S -> NP_obj[AGR={first_num}] "\
                            f"V_that[AGR={first_num}] NP[AGR={sec_num}]'*' "\
                            f"PP[AGR={third_num}] COMPL VP[AGR={sec_num}]'^' "
                    incorrect[f"{sec_num}_{first_num}_{third_num}"] =\
                            f"S -> NP_obj[AGR={first_num}] "\
                            f"V_that[AGR={first_num}] NP[AGR={sec_num}]'*' "\
                            f"PP[AGR={third_num}] COMPL VP[AGR={wrong_num}]'^' "\

    elif template == "that_nounpp_adv":
        for sec_num in conditions:
            wrong_num = get_opposite_number(sec_num)
            for first_num in conditions:
                for third_num in conditions:
                    # Mention the second number first, as we are tracking the number
                    # agreement for that noun
                    correct[f"{sec_num}_{first_num}_{third_num}"] =\
                            f"S -> NP_obj[AGR={first_num}]"\
                            f"V_that[AGR={first_num}] NP[AGR={sec_num}]'*' "\
                            f"PP[AGR={third_num}] ADV COMPL VP[AGR={sec_num}]'^'"
                    incorrect[f"{sec_num}_{first_num}_{third_num}"] =\
                            f"S -> NP_obj[AGR={first_num}]"\
                            f"V_that[AGR={first_num}] NP[AGR={sec_num}]'*' "\
                            f"PP[AGR={third_num}] ADV COMPL VP[AGR={wrong_num}]'^'"\

    elif template == "noun_conj":
        for subj1_num in conditions:
            for subj2_num in conditions:
                correct[f"{subj1_num}_{subj2_num}"] = f"S -> NP[AGR={subj1_num}]"\
                        f" 'en' NP[AGR={subj2_num}]'*' VP[AGR='pl']'^' COMPL"
                incorrect[f"{subj1_num}_{subj2_num}"] = f"S -> NP[AGR={subj1_num}]"\
                        f" 'en' NP[AGR={subj2_num}]'*' VP[AGR='sg']'^' COMPL"

    elif template == "s_conj":
        for subj1_num in conditions:
            for subj2_num in conditions:
                wrong_num = get_opposite_number(subj2_num)
                correct[f"{subj1_num}_{subj2_num}"] = f"S -> NP[AGR={subj1_num}]"\
                        f" VP[AGR={subj1_num}] 'en' NP[AGR={subj2_num}]'*' "\
                        f"VP[AGR={subj2_num}]'^' COMPL"
                incorrect[f"{subj1_num}_{subj2_num}"] = f"S -> NP[AGR={subj1_num}]"\
                        f" VP[AGR={subj1_num}] 'en' NP[AGR={subj2_num}]'*' "\
                        f"VP[AGR={wrong_num}]'^' COMPL"

    else:
        sys.exit("No valid template")

    return grammar, correct, incorrect
