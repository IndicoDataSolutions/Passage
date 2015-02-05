import sys
import theano
import theano.tensor as T
import numpy as np
from time import time

import costs
import updates
import iterators 
from utils import case_insensitive_import
from preprocessing import LenFilter, standardize_targets

def flatten(l):
    return [item for sublist in l for item in sublist]

class RNN(object):

    def __init__(self, layers, cost, updater='Adam', verbose=2, Y=T.matrix(), iterator='SortedPadded'):
        self.settings = locals()
        del self.settings['self']
        self.layers = layers

        if isinstance(cost, basestring):
            self.cost = case_insensitive_import(costs, cost)
        else:
            self.cost = cost

        if isinstance(updater, basestring):
            self.updater = case_insensitive_import(updates, updater)()
        else:
            self.updater = updater

        if isinstance(iterator, basestring):
            self.iterator = getattr(iterators, iterator)()
        else:
            self.iterator = iterator

        self.verbose = verbose
        for i in range(1, len(self.layers)):
            self.layers[i].connect(self.layers[i-1])
        self.params = flatten([l.params for l in layers])

        self.X = self.layers[0].input
        self.y_tr = self.layers[-1].output(dropout_active=True)
        self.y_te = self.layers[-1].output(dropout_active=False)
        self.Y = Y

        cost = self.cost(self.Y, self.y_tr)
        self.updates = self.updater.get_updates(self.params, cost)

        self._train = theano.function([self.X, self.Y], cost, updates=self.updates)
        self._cost = theano.function([self.X, self.Y], cost)
        self._predict = theano.function([self.X], self.y_te)

    def fit(self, trX, trY, batch_size=64, n_epochs=1, len_filter=LenFilter()):
        if len_filter is not None:
            trX, trY = len_filter.filter(trX, trY)
        trY = standardize_targets(trY, cost=self.cost)

        n = 0.
        stats = []
        t = time()
        costs = []
        for e in range(n_epochs):
            epoch_costs = []
            for xmb, ymb in self.iterator.iterXY(trX, trY):
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
            return self.predict_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.predict_idxs(X)
        else:
            raise NotImplementedError

    def predict_iterator(self, X):
        preds = []
        for xmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
        return np.vstack(preds)

    def predict_idxs(self, X):
        preds = []
        idxs = []
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
            idxs.extend(idxmb)
        return np.vstack(preds)[np.argsort(idxs)]