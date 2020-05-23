import pickle
import argparse
import pandas as pd
pd.options.display.max_rows = 1500


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

    if "that_nounpp" in args.input:
        df = pd.DataFrame(columns=["SSS", "SSP", "SPS", "SPP", "PSS", "PSP", "PPS", "PPP"])

        df.loc["-1"] = [0.998, 0.985, 1.0, 1.0, 1.0, 0.993, 1.0, 1.0]

        for k in list(data):
            column_data = []
            res = data[k]
            for x in ["singular", "plural"]:
                for y in ["singular", "plural"]:
                    for z in ["singular", "plural"]:
                        column_data.append(res[f"accuracy_{x}_{y}_{z}"])

            df.loc[k] = column_data[:8]

        print(df.loc[["-1"]])
        diff = df.loc[(df["SSS"] <= (df.loc["-1"]["SSS"] - 0.08))
                    | (df["SSP"] <= (df.loc["-1"]["SSP"] - 0.08))
                    | (df["SPS"] <= (df.loc["-1"]["SPS"] - 0.08))
                    | (df["SPP"] <= (df.loc["-1"]["SPP"] - 0.08))
                    | (df["PSS"] <= (df.loc["-1"]["PSS"] - 0.08))
                    | (df["PSP"] <= (df.loc["-1"]["PSP"] - 0.08))
                    | (df["PPS"] <= (df.loc["-1"]["PPS"] - 0.08))
                    | (df["PPP"] <= (df.loc["-1"]["PPP"] - 0.08))
                    ]

    elif "nounpp" in args.input:
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
                    | (df["PP"] < (df.loc["-1"]["PP"] - 0.1))
                    | (df["SP"] < (df.loc["-1"]["SP"] - 0.1))
                    | (df["SS"] < (df.loc["-1"]["SS"] - 0.1))]
        diff.append(df.loc["-1"])
    print(diff)
