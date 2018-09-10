import sys
import theano
import theano.tensor as T
import numpy as np
from time import time

import passage.costs as costs
import passage.updates as updates
import passage.iterators as iterators

from passage.utils import case_insensitive_import, save
from passage.preprocessing import LenFilter, standardize_targets

def flatten(l):
    return [item for sublist in l for item in sublist]

try:
    basestring
    BaseString = basestring
except NameError:
    BaseString = (str, bytes)

class RNN(object):

    def __init__(self, layers, cost, updater='Adam', verbose=2, Y=T.matrix(), iterator='SortedPadded'):
        self.settings = locals()
        del self.settings['self']
        self.layers = layers

        if isinstance(cost, BaseString):
            self.cost = case_insensitive_import(costs, cost)
        else:
            self.cost = cost

        if isinstance(updater, BaseString):
            self.updater = case_insensitive_import(updates, updater)()
        else:
            self.updater = updater

        if isinstance(iterator, BaseString):
            self.iterator = case_insensitive_import(iterators, iterator)()
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

    def fit(self, trX, trY, batch_size=64, n_epochs=1, len_filter=LenFilter(), snapshot_freq=1, path=None):
        """Train model on given training examples and return the list of costs after each minibatch is processed.

        Args:
          trX (list) -- Inputs
          trY (list) -- Outputs
          batch_size (int, optional) -- number of examples in a minibatch (default 64)
          n_epochs (int, optional)  -- number of epochs to train for (default 1)
          len_filter (object, optional) -- object to filter training example by length (default LenFilter())
          snapshot_freq (int, optional) -- number of epochs between saving model snapshots (default 1)
          path (str, optional) -- prefix of path where model snapshots are saved.
            If None, no snapshots are saved (default None)

        Returns:
          list -- costs of model after processing each minibatch
        """
        if len_filter is not None:
            trX, trY = len_filter.filter(trX, trY)
        trY = standardize_targets(trY, cost=self.cost)

        n = 0.
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
                print(status)
            if path and e % snapshot_freq == 0:
                save(self, "{0}.{1}".format(path, e))
        return costs

    def predict(self, X):
        if isinstance(self.iterator, iterators.Padded) or isinstance(self.iterator, iterators.Linear):
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
