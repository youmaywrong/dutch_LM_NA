import os
import torch

class Dictionary(object):
    def __init__(self, path=None):
        self.word2idx = {}
        self.idx2word = []
        if path:
            self.load(path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def load(self, path):
        with open(path, "r") as f:
            for line in f:
                self.add_word(line.rstrip("\n"))

    def save(self, path):
        with open(path, "w") as f:
            for w in self.idx2word:
                f.write(f"{w}\n")
