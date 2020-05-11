import sys, os, argparse
import torch
import data, lstm
import pickle, pandas
import time
import copy
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable
from predict import *

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="model.pt",
    help="Model (meta file) to use")
parser.add_argument("-i", "--input", type=str, required=True,
    help="Input sentences (tsv file)")
parser.add_argument("-o", "--output", type=str, default="output_ablation")
parser.add_argument("-v", "--vocabulary", type=str,
    default="data/vocabulary/vocab.txt",
    help="Vocabulary of the training corpus that the model was trained on")
parser.add_argument("-u", "--unit", type=int, default=-1,
    help="Network unit to ablate")
parser.add_argument("-s", "--seed", type=int, default=5,
    help="Random seed for adding random units")
parser.add_argument("--number_of_units", type=int, default=1300)
parser.add_argument("--eos", type=str, default="<eos>",
    help="Token that indicates end of sentence")
parser.add_argument("--unk", type=str, default="<unk>",
    help="Token that indicates an unknown token"),
parser.add_argument("--cuda", action="store_true", default=False)

args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)
template = args.input.split("/")[-1].replace(".tsv", "")
output_fn = f"{template}.info"

# Create Dictionary object from the vocabulary
vocab = data.Dictionary(args.vocabulary)
data = pandas.read_csv(args.input, sep="\t", header=0)
header = list(data)
sentences = data.loc[:, "agreement"]

model = load_model(args.model, args.cuda)

# If there are units to ablate
if args.unit > -1:
    if args.unit < 650:
        target_unit = torch.LongTensor(np.array([[int(args.unit)]]))
        if args.cuda:
            target_unit.cuda()
        model.rnn.weight_hh_l0.data[:, target_unit] = 0

    elif args.unit > 649 and args.unit < 1300:
        target_unit = torch.LongTensor(np.array([[int(args.unit)-650]]))
        if args.cuda:
            target_unit.cuda()
        model.rnn.weight_hh_l1.data[:, target_unit] = 0
        model.decoder.weight.data[:, target_unit] = 0

    else:
        sys.exit("Invalid unit number")

# Initial sentences are all . <eos>, feed these to the model
# (Do not start in the original state)
init_sentence = " ".join([". <eos>"] * 5)
hidden = model.init_hidden(1)
init_out, init_h = feed_sentence(model, hidden, init_sentence.split(" "), vocab,
    args.cuda)

log_p_targets_correct, log_p_targets_wrong =\
    get_predictions(data, sentences, model, init_out, init_h, vocab, args.cuda)
out = categorise_predictions(data, sentences, log_p_targets_correct,
    log_p_targets_wrong)

template = args.input.split("/")[-1].replace(".tsv", "")
info = {}

try:
    with open(os.path.join(args.output, output_fn), "rb") as f:
        info = pickle.load(f)
except Exception:
    pass

if args.unit > -1:
    info[str(args.unit)] = out
else:
    info = out

with open(os.path.join(args.output, output_fn), "wb") as f:
    pickle.dump(info, f, -1)

print(f"Information saved to {args.output}/{output_fn}\n")
