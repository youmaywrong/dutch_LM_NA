from results import read_results

for task in ["adv", "namepp", "noun_conj", "qnty_namepp", "qnty_nounpp", "nounpp",
            "qnty_simple", "rel_def_obj", "rel_def", "rel_nondef", "s_conj",
            "simple", "that_adv", "that_compl", "that_nounpp", "that"]:

    print(f"Reading results of {task}")
    res = read_results(f"output_ablation/{task}.info")

    if task in ["simple", "adv", "namepp", "qnty_simple", "qnty_namepp", "rel_def", "rel_nondef"]:
        print(f"S {res["873"]["accuracy_plural"]}")
        print(f"P {res["873"]["accuracy_singular"]}")

    elif task in ["nounpp", "noun_conj", "qnty_nounpp", "that", "that_adv", "that_compl", "rel_def_obj", "s_conj"]:
        print(f"SS {res["873"]["accuracy_singular_singular"]}")
        print(f"SP {res["873"]["accuracy_singular_plural"]}")
        print(f"PS {res["873"]["accuracy_plural_singular"]}")
        print(f"PP {res["873"]["accuracy_plural_plural"]}")

    elif task == "that_nounpp":
        print(f"SSS {res["873"]["accuracy_singular_singular_singular"]}")
        print(f"SSP {res["873"]["accuracy_singular_singular_plural"]}")
        print(f"SPS {res["873"]["accuracy_singular_plural_singular"]}")
        print(f"SPP {res["873"]["accuracy_singular_plural_plural"]}")
        print(f"PSS {res["873"]["accuracy_plural_singular_singular"]}")
        print(f"PSP {res["873"]["accuracy_plural_singular_plural"]}")
        print(f"PPS {res["873"]["accuracy_plural_plural_singular"]}")
        print(f"PPP {res["873"]["accuracy_plural_plural_plural"]}")
