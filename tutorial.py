import os

import pandas as pd
from sklearn import metrics

from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.model import RNN

def load_gender_data(ntrain=10000, ntest=10000):
	file_loc = os.path.dirname(os.path.realpath(__file__))
	relative_path = "data/blogger_data_2.csv"
	fullpath = os.path.join(file_loc, relative_path)
	data = pd.read_csv(fullpath, nrows=ntrain+ntest)
	X = data['text'].values
	X = [str(x) for x in X] #ugly nan cleaner
	Y = data['gender'].values
	trX = X[:-ntest]
	teX = X[-ntest:]
	trY = Y[:-ntest]
	teY = Y[-ntest:]
	return trX, teX, trY, teY

trX, teX, trY, teY = load_gender_data(ntrain=10000)

tokenizer = Tokenizer(min_df=10, max_features=50000)
print trX[:1]
trX = tokenizer.fit_transform(trX)
teX = tokenizer.transform(teX)
print tokenizer.inverse_transform(trX[:1])
print tokenizer.n_features

layers = [
	Embedding(size=256, n_features=tokenizer.n_features),
	GatedRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid', init='orthogonal', seq_output=False),
	Dense(size=1, activation='sigmoid', init='orthogonal')
]

model = RNN(layers=layers, cost='bce')
for i in range(100):
	model.fit(trX, trY, n_epochs=1)
	tr_preds = model.predict(trX[:len(teY)])
	te_preds = model.predict(teX)

	tr_acc = metrics.accuracy_score(trY[:len(teY)], tr_preds > 0.5)
	te_acc = metrics.accuracy_score(teY, te_preds > 0.5)

	print i, tr_acc, te_acc