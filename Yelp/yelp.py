import os
from torch.utils. data import Dataset

from utils import load_pickle


class Yelp(Dataset):

    def __init__(self, root, split, label=False):
        self.label = label
        self.data = load_pickle(os.path.join(root, "%s.pkl" % split))
        self._word_to_idx = load_pickle(os.path.join(root, 'vocab.pkl'))
        self._symbols = {'<pad>': self._word_to_idx['<pad>'],
                         '<unk>': self._word_to_idx['<unk>'],
                         '<bos>': self._word_to_idx['<bos>'],
                         '<eos>': self._word_to_idx['<eos>']}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label:
            return self.data[idx]
        enc_input, dec_input, target, length, label = self.data[idx]
        return enc_input, dec_input, target, length

    @property
    def vocab_size(self):
        return len(self._word_to_idx)

    @property
    def word_to_idx(self):
        return self._word_to_idx

    @property
    def idx_to_word(self):
        return {val: key for key, val in self._word_to_idx.items()}

    @property
    def symbols(self):
        return self._symbols
