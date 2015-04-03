import numpy as np

from utils import shuffle, iter_data
from theano_utils import floatX, intX

def padded(seqs):
    if isinstance(seqs[0][0], (int, float)):
        lens = map(len, seqs)
        max_len = max(lens)
    else:
        lens = [len(s[0]) for s in seqs]
        max_len = max(lens)
    seqs_padded = []
    for seq, seq_len in zip(seqs, lens):
        n_pad = max_len - seq_len
        if isinstance(seq[0], (int, float)):
            seq = [0] * n_pad + seq
        else:
            seq = seq[0]
            seq = [[0]*len(seq[0])] * n_pad + seq
            # print np.asarray(seq).shape
        seqs_padded.append(seq)
    seqs_padded = np.asarray(seqs_padded)
    shape = range(len(seqs_padded.shape))
    shape[0] = 1
    shape[1] = 0
    shape = tuple(shape)
    seqs_padded = seqs_padded.transpose(*shape)
    return seqs_padded

class Linear(object):
    """
    Useful for training on real valued data where first dimension is examples, 
    second dimension is to be iterated over, and third dimension is data vectors.

    size is the number of examples per minibatch
    shuffle controls whether or not the order of examples is shuffled before iterating over
    x_dtype is for casting input data
    y_dtype is for casting target data
    """

    def __init__(self, size=64, shuffle=True, x_dtype=floatX, y_dtype=floatX, y_seq=False):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.y_seq = y_seq

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            xmb = self.x_dtype(xmb)
            shape = range(len(xmb.shape))
            shape[0] = 1
            shape[1] = 0
            shape = tuple(shape)
            xmb = xmb.transpose(*shape)
            yield xmb

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for xmb, ymb in iter_data(X, Y, size=self.size):
            xmb = self.x_dtype(xmb)
            shape = range(len(xmb.shape))
            shape[0] = 1
            shape[1] = 0
            shape = tuple(shape)
            xmb = xmb.transpose(*shape)
            if self.y_seq:
                ymb = self.y_dtype(ymb)
                shape = range(len(ymb.shape))
                shape[0] = 1
                shape[1] = 0
                shape = tuple(shape)
                ymb = ymb.transpose(*shape)
            else:
                ymb = self.y_dtype(ymb)               
            yield xmb, ymb

class Padded(object):

    def __init__(self, size=64, shuffle=True, x_dtype=intX, y_dtype=floatX):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            xmb = padded(xmb)
            yield self.x_dtype(xmb)

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for xmb, ymb in iter_data(X, Y, size=self.size):
            xmb = padded(xmb)
            yield self.x_dtype(xmb), self.y_dtype(ymb)

class SortedPadded(object):

    def __init__(self, size=64, shuffle=True, x_pad=True, y_pad=False, x_dtype=intX, y_dtype=floatX):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.x_pad = x_pad
        self.y_pad = y_pad

    def iterX(self, X):
        for x_chunk, chunk_idxs in iter_data(X, np.arange(len(X)), size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            chunk_idxs = [chunk_idxs[idx] for idx in sort]
            for xmb, idxmb in iter_data(x_chunk, chunk_idxs, size=self.size):
                if self.x_pad:
                    xmb = padded(xmb)
                yield self.x_dtype(xmb), idxmb   

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for x_chunk, y_chunk in iter_data(X, Y, size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            y_chunk = [y_chunk[idx] for idx in sort]
            mb_chunks = [[x_chunk[idx:idx+self.size], y_chunk[idx:idx+self.size]] for idx in range(len(x_chunk))[::self.size]]
            mb_chunks = shuffle(mb_chunks)
            for xmb, ymb in mb_chunks:
                if self.x_pad:
                    xmb = padded(xmb)
                if self.y_pad:
                    ymb = padded(ymb)
                yield self.x_dtype(xmb), self.y_dtype(ymb)  