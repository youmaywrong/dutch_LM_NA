import argparse
import sys, os
import random
import csv, pandas
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template", required=True)
    parser.add_argument("-d", "--directory", required=True)
    parser.add_argument("-o", "--output", default="tasks")
    parser.add_argument("-n", "--number", default=600)
    args = parser.parse_args()

    path = f"{args.directory}/{args.template}.tsv"
    if not os.path.exists(path):
        sys.exit(f"{args.template}.tsv does not exist in {args.directory}.\n"\
                 f"Data for {args.template} can be generated with\n"\
                 f"python generate_tasks.py -t {args.template}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load data and convert all integers to strings
    data = pandas.read_csv(path, sep="\t", header=0).applymap(str)
    header = data.columns.values
    nums = sum("number" in col for col in header)

    separated_conditions, amounts = [], []
    options = ["singular", "plural"]

    if nums == 1:
        for num1 in options:
            curr_condition = data.loc[data["number1"] == num1]
            separated_conditions.append(curr_condition)
            amounts.append(min(args.number, len(curr_condition)))

    elif nums == 2:
        for num1 in options:
            for num2 in options:
                curr_condition = data.loc[(data["number1"] == num1) &\
                                    (data["number2"] == num2)]
                separated_conditions.append(curr_condition)
                amounts.append(min(args.number, len(curr_condition)))

    elif nums == 3:
        for num1 in options:
            for num2 in options:
                for num3 in options:
                    curr_condition = data.loc[(data["number1"] == num1) &\
                                        (data["number2"] == num2) &\
                                        (data["number3"] == num3)]
                    separated_conditions.append(curr_condition)
                    amounts.append(min(args.number, len(curr_condition)))

    # Make sure the dataset is balanced, i.e. same amount for each condition
    max_allowed_per_condition = min(amounts)
    print(f"Sampling {max_allowed_per_condition} sentences per condition for {args.template}")
    amounts = [max_allowed_per_condition] * len(amounts)
    header = list(data)
    random.seed()

    with open(os.path.join(args.output, f"{args.template}.tsv"), "w") as f:
        f.write("\t".join(header) + "\n")
        for c, a in zip(separated_conditions, amounts):
            sentences = c.values.tolist()
            sampled = random.sample(sentences, a)

            for s in sampled:
                f.write("\t".join(s) + "\n")
