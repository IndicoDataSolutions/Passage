import numpy as np

from passage.models import RNN
from passage.updates import NAG, Regularizer
from passage.layers import Generic, GatedRecurrent, Dense
from passage.utils import load, save

from load import load_mnist

trX, teX, trY, teY = load_mnist()

#Use generic layer - RNN processes a size 28 vector at a time scanning from left to right 
layers = [
	Generic(size=28),
	GatedRecurrent(size=512, p_drop=0.2),
	Dense(size=10, activation='softmax', p_drop=0.5)
]

#A bit of l2 helps with generalization, higher momentum helps convergence
updater = NAG(momentum=0.95, regularizer=Regularizer(l2=1e-4))

#Linear iterator for real valued data, cce cost for softmax
model = RNN(layers=layers, updater=updater, iterator='linear', cost='cce')
model.fit(trX, trY, n_epochs=20)

tr_preds = model.predict(trX[:len(teY)])
te_preds = model.predict(teX)

tr_acc = np.mean(trY[:len(teY)] == np.argmax(tr_preds, axis=1))
te_acc = np.mean(teY == np.argmax(te_preds, axis=1))

# Test accuracy should be between 98.9% and 99.3%
print 'train accuracy', tr_acc, 'test accuracy', te_acc