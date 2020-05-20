import sys, os
import torch
import argparse
import data, lstm
import numpy as np
import pickle, pandas
import copy

from tqdm import tqdm
from torch.autograd import Variable

def feed_input(model, hidden, word, vocab, cuda):
    input = torch.autograd.Variable(torch.LongTensor([[vocab.word2idx[word]]]))
    if cuda:
        input = input.cuda()
    output, hidden = model(input, hidden)
    return output, hidden

def feed_sentence(model, hidden, sentence, vocab, cuda):
    outputs = []
    for word in sentence:
        out, hidden = feed_input(model, hidden, word, vocab, cuda)
        outputs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outputs, hidden


def load_model(model_file, cuda):
    # Load model
    model = torch.load(model_file, map_location=lambda storage, loc: storage)
    if cuda:
        model.cuda()
    model.rnn.flatten_parameters()

    # Send extra argument with model parameters to forward function
    model.rnn.forward = lambda input, hidden: lstm.forward(model.rnn, input, hidden)
    model_original = copy.deepcopy(model.state_dict())
    model.load_state_dict(model_original)

    return model


def get_predictions(data, sentences, model, init_out, init_h, vocab, cuda):
    # Initialise log probabilities at 0
    log_p_targets_correct = np.zeros((len(sentences), 1))
    log_p_targets_wrong = np.zeros((len(sentences), 1))

    for i, sentence in enumerate(tqdm(sentences)):
        sentence = sentence.split(" ")
        out = None
        hidden = init_h

        for j, token in enumerate(sentence):
            # Unknown word
            if token not in vocab.word2idx:
                token = "<unk>"

            input = Variable(torch.LongTensor([[vocab.word2idx[token]]]))
            if cuda:
                input = input.cuda()

            out, hidden = model(input, hidden)
            out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)
            #
            if j == data.loc[i, "verb_index"] - 1:
                log_p_targets_correct[i] = out[0, 0, vocab.word2idx[data.loc[i, "correct_verb"]]].data.item()
                log_p_targets_wrong[i] = out[0, 0, vocab.word2idx[data.loc[i, "incorrect_verb"]]].data.item()

    return log_p_targets_correct, log_p_targets_wrong


def categorise_predictions(data, sentences, log_p_targets_correct, log_p_targets_wrong):
    nums = sum("number" in col for col in list(data))
    options = ["singular", "plural"]

    correct = log_p_targets_correct > log_p_targets_wrong
    score_on_task = np.sum(correct)
    p_difference = np.exp(log_p_targets_correct) - np.exp(log_p_targets_wrong)
    score_on_task_p_difference = np.mean(p_difference)
    score_on_task_p_difference_std = np.std(p_difference)
    info = {
        'log_p_targets_correct': log_p_targets_correct,
        'log_p_targets_wrong': log_p_targets_wrong,
        'score_on_task': score_on_task,
        'accuracy_score_on_task': score_on_task,
    }

    if nums == 1:
        for num1 in options:
            # Get indices corresponding to the rows of this condition
            relevant_idx = np.array(data.index[data["number1"] == num1])
            scores = correct.flatten()[relevant_idx]
            info[f"accuracy_{num1}"] = np.mean(scores)
            print(f"accuracy for {num1}: {np.mean(scores) * 100}")

    elif nums == 2:
        for num1 in options:
            for num2 in options:
                relevant_idx = np.array(data.index[(data["number1"] == num1) &\
                                            (data["number2"] == num2)])
                scores = correct.flatten()[relevant_idx]
                info[f"accuracy_{num1}_{num2}"] = np.mean(scores)
                print(f"accuracy for {num1}_{num2}: {np.mean(scores) * 100}")

    elif nums == 3:
        for num1 in options:
            for num2 in options:
                for num3 in options:
                    relevant_idx = np.array(data.index[(data["number1"] == num1) &\
                                                (data["number2"] == num2) &\
                                                (data["number3"] == num3)])
                    scores = correct.flatten()[relevant_idx]
                    info[f"accuracy_{num1}_{num2}_{num3}"] = np.mean(scores)
                    print(f"accuracy for {num1}_{num2}_{num3}: {np.mean(scores) * 100}")

    print('accuracy: ' + str(100*score_on_task/len(sentences)))
    print('p_difference: %1.3f +- %1.3f' % (score_on_task_p_difference, score_on_task_p_difference_std))

    return info
