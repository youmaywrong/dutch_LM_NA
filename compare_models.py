import pandas as pd

task = ["adv", "namepp", "noun_conj", "nounpp", "qnty_namepp",
      "qnty_nounpp", "qnty_simple", "rel_def", "rel_def_obj",
      "rel_nondef", "s_conj", "simple", "that_adv", "that", "that_compl",
      "that_nounpp"]

cols = []
for i in task:
    test = pd.read_pickle(f"output_23/{i}.info")
    cols.extend([c for c in list(test) if "accuracy" in c and c not in cols])

df = pd.DataFrame(columns=cols)
for i in task:
    test = pd.read_pickle(f"output_23/{i}.info")
    df.append(test, ignore_index=True)

print(df)
# test = pd.read_pickle("output_23/that_adv.info")
# print(test)
# print(list(test))
