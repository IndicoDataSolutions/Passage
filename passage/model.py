import sys
import theano
import theano.tensor as T
import numpy as np
from time import time

import costs
import updates
import iterators 
from iterators import iter_data, padded
from utils import case_insensitive_import
from preprocessing import LenFilter, standardize_targets

def flatten(l):
    return [item for sublist in l for item in sublist]

class RNN(object):

    def __init__(self, layers, cost, updater='Adam', verbose=2):
        self.layers = layers

        if isinstance(cost, basestring):
            self.cost = case_insensitive_import(costs, cost)
        else:
            self.cost = cost

        if isinstance(updater, basestring):
            updater = case_insensitive_import(updates, updater)()
        else:
            updater = updater

        self.verbose = verbose
        for i in range(1, len(self.layers)):
            self.layers[i].connect(self.layers[i-1])
        self.params = flatten([l.params for l in layers])

        self.X = self.layers[0].input
        self.y_tr = self.layers[-1].output(dropout_active=True)
        self.y_te = self.layers[-1].output(dropout_active=False)
        self.Y = T.matrix()

        cost = self.cost(self.Y, self.y_tr)
        self.updates = updater.get_updates(self.params, cost)

        self._train = theano.function([self.X, self.Y], cost, updates=self.updates)
        self._cost = theano.function([self.X, self.Y], cost)
        self._predict = theano.function([self.X], self.y_te)

    def fit(self, trX, trY, batch_size=64, n_epochs=1, len_filter=LenFilter(), iterator='SortedPadded'):
        self.batch_size = batch_size
        if len_filter is not None:
            trX, trY = len_filter.filter(trX, trY)
        trY = standardize_targets(trY, cost=self.cost)
        self.iterator = getattr(iterators, iterator)(trX, trY, size=batch_size)

        n = 0.
        stats = []
        t = time()
        costs = []
        for e in range(n_epochs):
            epoch_costs = []
            for xmb, ymb in self.iterator.iter():
                c = self._train(xmb, ymb)
                epoch_costs.append(c)
                n += len(ymb)
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = len(trY) - n % len(trY)
                    time_left = n_left/n_per_sec
                    sys.stdout.write("\rEpoch %d Seen %d samples Avg cost %0.4f Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)

            status = "Epoch %d Seen %d samples Avg cost %0.4f Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status
        return costs

    def predict(self, X):
        if isinstance(self.iterator, iterators.Padded):
            return self.predict_padded(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.predict_sorted_padded(X)
        else:
            raise NotImplementedError

    def predict_padded(self, X):
        preds = []
        for xmb in iter_data(X, size=self.batch_size):
            xmb = padded(xmb)
            pred = self._predict(xmb)
            preds.append(pred)
        return np.vstack(preds)

    def predict_sorted_padded(self, X):
        preds = []
        idxs = []
        for x_chunk, chunk_idxs in iter_data(X, np.arange(len(X)), size=self.batch_size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            idxs.extend(chunk_idxs[sort])
            for xmb in iter_data(x_chunk, size=self.batch_size):
                xmb = padded(xmb)
                pred = self._predict(xmb)
                preds.append(pred)
        return np.vstack(preds)[np.argsort(idxs)]