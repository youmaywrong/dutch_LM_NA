import pickle
import argparse
import pandas as pd

def read_results(filename):
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            return data
    except:
        print(f"{filename} is not readable.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
        help="Path to file")
    args = parser.parse_args()

    data = read_results(args.input)

    df = pd.DataFrame(columns=["SS", "SP", "PS", "PP"])
    for k in list(data):
        res = data[k]
        c1 = res["accuracy_singular_singular"]
        c2 = res["accuracy_singular_plural"]
        c3 = res["accuracy_plural_singular"]
        c4 = res["accuracy_plural_plural"]
        df.loc[k] = [c1, c2, c3, c4]

    df.loc["-1"] = [0.9983, 0.9783, 0.9467, 0.9450]

    print(df.loc[["-1"]])
    diff = df.loc[(df["PS"] < (df.loc["-1"]["PS"] - 0.1))
                or (df["PP"] < (df.loc["-1"]["PP"] - 0.1))
                or (df["SP"] < (df.loc["-1"]["SP"] - 0.1))
                or (df["SS"] < (df.loc["-1"]["SS"] - 0.1))]
    diff.append(df.loc["-1"])
    print(diff)
