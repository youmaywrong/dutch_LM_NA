import pandas as pd
header = f"agreement\tdisagreement\tcorrect_verb\tincorrect_verb\t"\
         f"subject_index\tverb_index\tnumber1\tnumber2\n"

original_data = pd.read_csv("lakretz_nounpp.txt", sep="\t", header=None)

agreement = original_data[1].tolist()
correct_verb = [sentence.split(' ')[-1] for sentence in agreement]
incorrect_verb = original_data[4].tolist()
disagreement = []
for (sentence, w) in zip(agreement, incorrect_verb):
    incorrect_sentence = sentence.split(' ')[:-1]
    incorrect_sentence.append(w)
    disagreement.append(" ".join(incorrect_sentence))
subject_index = [str(1)] * len(agreement)
verb_index = [str(5)] * len(agreement)
number1 = original_data[2].tolist()
number2 = original_data[3].tolist()


with open("lakretz_nounpp_reformatted.tsv", "w") as f:
    f.write(header)
    for line in zip(agreement, disagreement, correct_verb, incorrect_verb, subject_index, verb_index, number1, number2):
        f.write("\t".join(line) + "\n")
