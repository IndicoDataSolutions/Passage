import theano
import theano.tensor as T

softmax = T.nnet.softmax
rectify = lambda x: (x + abs(x)) / 2.0
tanh = T.tanh
sigmoid = T.nnet.sigmoid
linear = lambda x: x
t_rectify = lambda x: x * (x > 1)
t_linear = lambda x: x * (abs(x) > 1)
maxout = lambda x: T.maximum(x[:, 0::2], x[:, 1::2])
clipped_maxout = lambda x: T.clip(T.maximum(x[:, 0::2], x[:, 1::2]), -1., 1.)
clipped_rectify = lambda x: T.clip((x + abs(x)) / 2.0, 0., 1.)
hard_tanh = lambda x: T.clip(x, -1. , 1.)
steeper_sigmoid = lambda x: 1./(1. + T.exp(-3.75 * x))
hard_sigmoid = lambda x: T.clip(x + 0.5, 0., 1.)