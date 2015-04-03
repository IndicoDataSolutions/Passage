import theano
import theano.tensor as T

def SequenceCategoricalCrossEntropy(y_true, y_pred):
	if y_true.dtype == 'int32':
		y_true = T.flatten(y_true, outdim=1)
	else:
		y_true = y_true.reshape((y_true.shape[0]*y_true.shape[1], -1))
	y_pred = y_pred.reshape((y_true.shape[0]*y_true.shape[1], -1))
	return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

def CategoricalCrossEntropy(y_true, y_pred):
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

def BinaryCrossEntropy(y_true, y_pred):
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()

def MeanSquaredError(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()

def MeanAbsoluteError(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean()

def SquaredHinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()

def Hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean()

seq_cce = SeqCCE = SequenceCategoricalCrossEntropy
cce = CCE = CategoricalCrossEntropy
bce = BCE = BinaryCrossEntropy
mse = MSE = MeanSquaredError
mae = MAE = MeanAbsoluteError
