import math
import numpy as np
from collections import Counter

from utils import floatX, intX, shuffle, iter_data

def padded(seqs):
    lens = map(len, seqs)
    max_len = max(lens)
    seqs_padded = []
    for seq, seq_len in zip(seqs, lens):
        n_pad = max_len - seq_len 
        seq = [0] * n_pad + seq
        seqs_padded.append(seq)
    return intX(seqs_padded).transpose(1, 0)

class Padded(object):

    def __init__(self, seqs, targets, size=64, shuffle=True):
        self.seqs = seqs
        self.targets = targets
        self.size = size
        self.shuffle = shuffle

    def iter(self):
        
        if self.shuffle:
            self.seqs, self.targets = shuffle(self.seqs, self.targets)

        for i in range(0, len(self.seqs), self.size):
            xmb, ymb = self.seqs[i:i+self.size], self.targets[i:i+self.size]
            xmb = padded(xmb)
            ymb = floatX(ymb)
            yield xmb, ymb

class SortedPadded(object):

    def __init__(self, seqs, targets, size=64, shuffle=True):
        self.seqs = seqs
        self.targets = targets
        self.size = size
        self.shuffle = shuffle

    def iter(self):
        
        if self.shuffle:
            self.seqs, self.targets = shuffle(self.seqs, self.targets)

        for x_chunk, y_chunk in iter_data(self.seqs, self.targets, size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            y_chunk = [y_chunk[idx] for idx in sort]
            # print range(len(x_chunk))[::self.size]
            mb_chunks = [[x_chunk[idx:idx+self.size], y_chunk[idx:idx+self.size]] for idx in range(len(x_chunk))[::self.size]]
            mb_chunks = shuffle(mb_chunks)
            for xmb, ymb in mb_chunks:
                xmb = padded(xmb)
                # print xmb.shape
                ymb = floatX(ymb)
                yield xmb, ymb