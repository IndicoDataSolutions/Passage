import numpy as np
import theano
import theano.tensor as T
import string
from collections import Counter

def one_hot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

def flatten(l):
    return [item for sublist in l for item in sublist]

def lbf(l,b):
    return [el for el, condition in zip(l, b) if condition]

def list_index(l, idxs):
    return [l[idx] for idx in idxs]

def tokenize(text):
    punctuation = set(string.punctuation)
    punctuation.add('\n')
    punctuation.add('\t')
    punctuation.add('')
    tokenized = []
    w = ''
    for t in text:
        if t in punctuation:
            tokenized.append(w)
            tokenized.append(t)
            w = ''
        elif t == ' ':
            tokenized.append(w)
            w = ''
        else:
            w += t
    if w != '':
        tokenized.append(w)
    tokenized = [token for token in tokenized if token]
    return tokenized

def token_encoder(texts, max_features=9997, min_df=10):
    df = {}
    for text in texts:
        tokens = set(text)
        for token in tokens:
            if token in df:
                df[token] += 1
            else:
                df[token] = 1
    k, v = df.keys(), np.asarray(df.values())
    valid = v >= min_df
    k = lbf(k, valid)
    v = v[valid]
    sort_mask = np.argsort(v)[::-1]
    k = list_index(k, sort_mask)[:max_features]
    v = v[sort_mask][:max_features]
    xtoi = dict(zip(k, range(3, len(k)+3)))
    return xtoi

def standardize_targets(Y, cost):
    Y = np.asarray(Y)
    ndim = len(Y.shape)
    if ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.shape[1] == 1 and cost.__name__ == 'CategoricalCrossEntropy':
        Y = one_hot(Y, negative_class=0.)
    if Y.shape[1] == 1 and 'Hinge' in cost.__name__:
        if len(np.unique(Y)) > 2:
            Y = one_hot(Y, negative_class=-1.)
        else:
            Y[Y==0] -= 1
    return Y

class Tokenizer(object):
    """
    For converting lists of text into tokens used by Passage models.
    max_features sets the maximum number of tokens (all others are mapped to UNK)
    min_df sets the minimum number of documents a token must appear in to not get mapped to UNK
    lowercase controls whether the text is lowercased or not
    character sets whether the tokenizer works on a character or word level

    Usage:
    >>> from passage.preprocessing import Tokenizer
    >>> example_text = ['This. is.', 'Example TEXT', 'is text']
    >>> tokenizer = Tokenizer(min_df=1, lowercase=True, character=False)
    >>> tokenized = tokenizer.fit_transform(example_text)
    >>> tokenized
    [[7, 5, 3, 5], [6, 4], [3, 4]]
    >>> tokenizer.inverse_transform(tokenized)
    ['this . is .', 'example text', 'is text']
    """

    def __init__(self, max_features=9997, min_df=10, lowercase=True, character=False):
        self.max_features = max_features
        self.min_df = min_df
        self.lowercase = lowercase
        self.character = character

    def fit(self, texts):
        if self.lowercase:
            texts = [text.lower() for text in texts]
        if self.character:
            tokens = [list(text) for text in texts]
        else:
            tokens = [tokenize(text) for text in texts]
        self.encoder = token_encoder(tokens, max_features=self.max_features-3, min_df=self.min_df)
        self.encoder['PAD'] = 0
        self.encoder['END'] = 1
        self.encoder['UNK'] = 2
        self.decoder = dict(zip(self.encoder.values(), self.encoder.keys()))
        self.n_features = len(self.encoder)
        return self

    def transform(self, texts):
        if self.lowercase:
            texts = [text.lower() for text in texts]
        if self.character:
            texts = [list(text) for text in texts]
        else:
            texts = [tokenize(text) for text in texts]
        tokens = [[self.encoder.get(token, 2) for token in text] for text in texts]
        return tokens

    def fit_transform(self, texts):
        self.fit(texts)
        tokens = self.transform(texts)
        return tokens

    def inverse_transform(self, codes):
        if self.character:
            joiner = ''
        else:
            joiner = ' '
        return [joiner.join([self.decoder[token] for token in code]) for code in codes]

class LenFilter(object):

    def __init__(self, max_len=1000, min_max_len=100, percentile=99):
        self.max_len = max_len
        self.percentile = percentile
        self.min_max_len = min_max_len

    def filter(self, *data):
        lens = [len(seq) for seq in data[0]]
        if self.percentile > 0:
            max_len = np.percentile(lens, self.percentile)
        max_len = np.clip(max_len, self.min_max_len, self.max_len)
        valid_idxs = [i for i, l in enumerate(lens) if l <= max_len]
        if len(data) == 1:
            return list_index(data[0], valid_idxs)
        else:
            return tuple([list_index(d, valid_idxs) for d in data])

