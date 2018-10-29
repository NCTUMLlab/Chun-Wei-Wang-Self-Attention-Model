import os
import numpy as np
from collections import Counter

from utils import save_pickle


root = 'data'
max_len = 64

print("Building vocabulary from Yelp data")
counter = Counter()
with open(os.path.join(root, 'train.txt')) as f:
    for line in f:
        words = line.strip().split()[1:] # drop label
        counter.update(words)

word_to_idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
for word, _ in counter.items():
    if word not in word_to_idx:
        word_to_idx[word] = len(word_to_idx)

# exclude <bos> and <pad> symbols
print("Vocabulary size: %d" % (len(word_to_idx) - 2))
save_pickle(word_to_idx, os.path.join(root, 'vocab.pkl'))

splits = ['train', 'valid', 'test']
num_sents, num_words = 0, 0
func = lambda seq: np.array([
    word_to_idx.get(symbol, word_to_idx['<unk>']) for symbol in seq])
for split in splits:
    print("Creating %s Yelp data" % split)
    data = []
    with open(os.path.join(root, "%s.txt" % split)) as f:
        for line in f:
            words = line.strip().split()[:max_len + 1]
            label, words = int(words[0]), words[1:]
            length = len(words)
            paddings = ['<pad>'] * (max_len - length)
            enc_input = func(words + paddings)
            dec_input = func(['<bos>'] + words + paddings)
            target = func(words + ['<eos>'] + paddings)
            data.append((enc_input, dec_input, target, length, label))
            num_words += length
    print("%s samples: %d" %(split.capitalize(), len(data)))
    save_pickle(data, os.path.join(root, "%s.pkl" % split))
    num_sents += len(data)

print("Average length: %.2f" %(num_words / num_sents))
