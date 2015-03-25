import numpy as np
import pandas as pd

from sklearn import metrics

from passage.layers import Generic, GatedRecurrent, Dense
from passage.models import RNN

from foxhound.utils.load import mnist

trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)

layers = [
	Generic(size=28, n_dim=3),
	GatedRecurrent(size=512, p_drop=0.2),
	Dense(size=10, activation='softmax', p_drop=0.5)
]

model = RNN(layers=layers, updater='nag', iterator='linear', cost='cce')

for i in range(100):
	model.fit(trX, trY)

	tr_preds = model.predict(trX[:len(teY)])
	print tr_preds.shape
	te_preds = model.predict(teX)
	print te_preds.shape

	tr_acc = metrics.accuracy_score(np.argmax(trY[:len(teY)], axis=1), np.argmax(tr_preds, axis=1))
	te_acc = metrics.accuracy_score(np.argmax(teY, axis=1), np.argmax(te_preds, axis=1))

	print i, tr_acc, te_acc