import sys, os
import torch
import argparse
import data, lstm
import numpy as np
import pickle
import pandas
import copy
from tqdm import tqdm
from torch.autograd import Variable

parser = argparse.ArgumentParser(
    description="LSTM Language Model Evaluation")
parser.add_argument("-m", "--model", type=str, default="model.pt",
    help="Model (meta file) to use")
parser.add_argument("-i", "--input", type=str, required=True,
    help="Input sentences (NA task)")
parser.add_argument("-v", "--vocabulary", type=str,
    default="vocabulary/vocab.txt", help="Vocabulary of the training corpus")
parser.add_argument("-o", "--output", default="output",
    help="Destination for the output")
parser.add_argument("--eos", default="<eos>", help="End-of-sentence token")
parser.add_argument("--unk", default="<unk>", help="Unknown-word token")
parser.add_argument("--cuda", action="store_true", default=False)

args = parser.parse_args()
if not os.path.exists(args.output):
    os.makedirs(args.output)

def feed_input(model, hidden, w):
    inp = torch.autograd.Variable(torch.LongTensor([[vocab.word2idx[w]]]))
    if args.cuda:
        inp = inp.cuda()
    out, hidden = model(inp, hidden)
    return out, hidden
def feed_sentence(model, h, sentence):
    outs = []
    for w in sentence:
        out, h = feed_input(model, h, w)
        outs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outs, h

vocab = data.Dictionary(args.vocabulary)
data = pandas.read_csv(f"{args.input}.tsv", sep="\t", header=0)
sentences = data.loc[:, "agreement"]

# correct, incorrect = [], []
# for i in range(len(sentences)):
#     correct.append(data.loc[i, "agreement"].split(" ")[data.loc[i, "verb_index"]])
#     incorrect.append(data.loc[i, "disagreement"].split(" ")[data.loc[i, "verb_index"]])
# Load model
print(f"Loading model {args.model}")
model = torch.load(args.model, map_location=lambda storage, loc: storage)
if args.cuda:
    model.cuda()
model.rnn.flatten_parameters()
# Hack the forward function to send an extra argument with model parameters
model.rnn.forward = lambda input, hidden: lstm.forward(model.rnn, input, hidden)
model_orig_state = copy.deepcopy(model.state_dict())

log_p_targets_correct = np.zeros((len(sentences), 1))
log_p_targets_wrong = np.zeros((len(sentences), 1))

model.load_state_dict(model_orig_state)
init_sentence = " ".join([". <eos>"] * 5)

output_fn = f"{args.output}.abl"

hidden = model.init_hidden(1)
init_out, init_h = feed_sentence(model, hidden, init_sentence.split(" "))

for i, sentence in enumerate(tqdm(sentences)):
    sentence = sentence.split(" ")
    out = None
    hidden = init_h

    for j, token in enumerate(sentence):
        if token not in vocab.word2idx:
            token = args.unk
        inp = Variable(torch.LongTensor([[vocab.word2idx[token]]]))
        if args.cuda:
            inp = inp.cuda()
        out, hidden = model(inp, hidden)
        out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)
        if j == data.loc[i, "verb_index"] - 1:
            log_p_targets_correct[i] = out[0, 0, vocab.word2idx[data.loc[i, "correct_verb"]]].data.item()
            log_p_targets_wrong[i] = out[0, 0, vocab.word2idx[data.loc[i, "incorrect_verb"]]].data.item()

score_on_task = np.sum(log_p_targets_correct > log_p_targets_wrong)
p_difference = np.exp(log_p_targets_correct) - np.exp(log_p_targets_wrong)
score_on_task_p_difference = np.mean(p_difference)
score_on_task_p_difference_std = np.std(p_difference)
out = {
    'log_p_targets_correct': log_p_targets_correct,
    'log_p_targets_wrong': log_p_targets_wrong,
    'score_on_task': score_on_task,
    'accuracy_score_on_task': score_on_task,
    'sentences': sentences,
    'num_sentences': len(sentences)
}
print('\naccuracy: ' + str(100*score_on_task/len(sentences)) + '%\n')
print('p_difference: %1.3f +- %1.3f' % (score_on_task_p_difference, score_on_task_p_difference_std))

with open(output_fn, 'wb') as fout:
    pickle.dump(out, fout, -1)
