import theano
import theano.tensor as T

def softmax(x):
    e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def rectify(x):
	return (x + abs(x)) / 2.0

def tanh(x):
	return T.tanh(x)

def sigmoid(x):
	return T.nnet.sigmoid(x)

def linear(x):
	return x

def t_rectify(x):
	return x * (x > 1)

def t_linear(x):
	return x * (abs(x) > 1)

def maxout(x):
	return T.maximum(x[:, 0::2], x[:, 1::2])

def conv_maxout(x):
	return T.maximum(x[:, 0::2, :, :], x[:, 1::2, :, :])

def clipped_maxout(x):
	return T.clip(T.maximum(x[:, 0::2], x[:, 1::2]), -5., 5.)

def clipped_rectify(x):
	return T.clip((x + abs(x)) / 2.0, 0., 5.)

def hard_tanh(x):
	return T.clip(x, -1. , 1.)

def steeper_sigmoid(x):
	return 1./(1. + T.exp(-3.75 * x))

def hard_sigmoid(x):
	return T.clip(x + 0.5, 0., 1.)