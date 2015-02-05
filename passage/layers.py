import theano
import theano.tensor as T
from theano.tensor.extra_ops import repeat
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano_utils import shared0s, floatX
import activations
import inits

import numpy as np

def dropout(X, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

def theano_one_hot(idx, n):
    z = T.zeros((idx.shape[0], n))
    one_hot = T.set_subtensor(z[T.arange(idx.shape[0]), idx], 1)
    return one_hot

srng = RandomStreams()

class Embedding(object):

    def __init__(self, size=128, n_features=256, init='uniform', weights=None):
        self.settings = locals()
        del self.settings['self']
        self.init = getattr(inits, init)
        self.size = size
        self.n_features = n_features
        self.input = T.imatrix()
        self.wv = self.init((self.n_features, self.size))
        self.params = [self.wv]

        if weights is not None:
            for param, weight in zip(self.params, weights):
                param.set_value(floatX(weight))

    def output(self, dropout_active=False):
        return self.wv[self.input]

class OneHot(object):

    def __init__(self, n_features, weights=None):
        self.settings = locals()
        del self.settings['self']
        self.size = n_features
        self.n_features = n_features
        self.input = T.imatrix()
        self.params = []

    def output(self, dropout_active=False):
        return theano_one_hot(self.input.flatten(), self.n_features).reshape((self.input.shape[0], self.input.shape[1], self.size))

class SimpleRecurrent(object):

    def __init__(self, size=256, activation='tanh', init='orthogonal', truncate_gradient=-1, seq_output=False, p_drop=0., weights=None):
        self.settings = locals()
        del self.settings['self']
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.init = getattr(inits, init)
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.p_drop = p_drop
        self.weights = weights

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size
        self.h0 = shared0s((1, self.size))
        if 'maxout' in self.activation_str:
            self.w_in = self.init((self.n_in, self.size*2))
            self.b_in = shared0s((self.size*2))
            self.w_rec = self.init((self.size, self.size*2))
        else:
            self.w_in = self.init((self.n_in, self.size))
            self.b_in = shared0s((self.size))
            self.w_rec = self.init((self.size, self.size))
        self.params = [self.h0, self.w_in, self.b_in, self.w_rec]
        
        if self.weights is not None:
            for param, weight in zip(self.params, self.weights):
                param.set_value(floatX(weight))    

    def step(self, x_t, h_tm1, w):
        h_t = self.activation(x_t + T.dot(h_tm1, w))
        return h_t

    def output(self, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        x_in = T.dot(X, self.w_in) + self.b_in
        out, _ = theano.scan(self.step,
            sequences=[x_in],
            outputs_info=[repeat(self.h0, x_in.shape[1], axis=0)],
            non_sequences=[self.w_rec],
            truncate_gradient=self.truncate_gradient
        )
        if self.seq_output:
            return out
        else:
            return out[-1]

class LstmRecurrent(object):

    def __init__(self, size=256, activation='tanh', gate_activation='steeper_sigmoid', init='orthogonal', truncate_gradient=-1, seq_output=False, p_drop=0., weights=None):
        self.settings = locals()
        del self.settings['self']
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.gate_activation = getattr(activations, gate_activation)
        self.init = getattr(inits, init)
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.p_drop = p_drop
        self.weights = weights

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size

        self.w_i = self.init((self.n_in, self.size))
        self.w_f = self.init((self.n_in, self.size))
        self.w_o = self.init((self.n_in, self.size))
        self.w_c = self.init((self.n_in, self.size))

        self.b_i = shared0s((self.size))
        self.b_f = shared0s((self.size))
        self.b_o = shared0s((self.size))
        self.b_c = shared0s((self.size))

        self.u_i = self.init((self.size, self.size))
        self.u_f = self.init((self.size, self.size))
        self.u_o = self.init((self.size, self.size))
        self.u_c = self.init((self.size, self.size))

        self.params = [self.w_i, self.w_f, self.w_o, self.w_c, 
            self.u_i, self.u_f, self.u_o, self.u_c,  
            self.b_i, self.b_f, self.b_o, self.b_c]

        if self.weights is not None:
            for param, weight in zip(self.params, self.weights):
                param.set_value(floatX(weight))    

    def step(self, xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1, u_i, u_f, u_o, u_c):
        i_t = self.gate_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.gate_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.gate_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def output(self, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        x_i = T.dot(X, self.w_i) + self.b_i
        x_f = T.dot(X, self.w_f) + self.b_f
        x_o = T.dot(X, self.w_o) + self.b_o
        x_c = T.dot(X, self.w_c) + self.b_c
        [out, cells], _ = theano.scan(self.step, 
            sequences=[x_i, x_f, x_o, x_c], 
            outputs_info=[T.alloc(0., X.shape[1], self.size), T.alloc(0., X.shape[1], self.size)], 
            non_sequences=[self.u_i, self.u_f, self.u_o, self.u_c],
            truncate_gradient=self.truncate_gradient
        )
        if self.seq_output:
            return out
        else:
            return out[-1]

class GatedRecurrent(object):

    def __init__(self, size=256, activation='tanh', gate_activation='steeper_sigmoid', init='orthogonal', truncate_gradient=-1, seq_output=False, p_drop=0., weights=None):
        self.settings = locals()
        del self.settings['self']   
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.gate_activation = getattr(activations, gate_activation)
        self.init = getattr(inits, init)
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.p_drop = p_drop
        self.weights = weights

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size
        self.h0 = shared0s((1, self.size))

        self.w_z = self.init((self.n_in, self.size))
        self.w_r = self.init((self.n_in, self.size))

        self.u_z = self.init((self.size, self.size))
        self.u_r = self.init((self.size, self.size))

        self.b_z = shared0s((self.size))
        self.b_r = shared0s((self.size))

        if 'maxout' in self.activation_str:
            self.w_h = self.init((self.n_in, self.size*2)) 
            self.u_h = self.init((self.size, self.size*2))
            self.b_h = shared0s((self.size*2))
        else:
            self.w_h = self.init((self.n_in, self.size)) 
            self.u_h = self.init((self.size, self.size))
            self.b_h = shared0s((self.size))   

        self.params = [self.h0, self.w_z, self.w_r, self.w_h, self.u_z, self.u_r, self.u_h, self.b_z, self.b_r, self.b_h]

        if self.weights is not None:
            for param, weight in zip(self.params, self.weights):
                param.set_value(floatX(weight))    


    def step(self, xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t
        return h_t

    def output(self, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], 
            outputs_info=[repeat(self.h0, x_h.shape[1], axis=0)], 
            non_sequences=[self.u_z, self.u_r, self.u_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.seq_output:
            return out
        else:
            return out[-1]  

class Dense(object):
    def __init__(self, size=256, activation='rectify', init='orthogonal', p_drop=0., weights=None):
        self.settings = locals()
        del self.settings['self']
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.init = getattr(inits, init)
        self.size = size
        self.p_drop = p_drop
        self.weights = weights

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size
        if 'maxout' in self.activation_str:
            self.w = self.init((self.n_in, self.size*2))
            self.b = shared0s((self.size*2))
        else:
            self.w = self.init((self.n_in, self.size))
            self.b = shared0s((self.size))
        self.params = [self.w, self.b]
        
        if self.weights is not None:
            for param, weight in zip(self.params, self.weights):
                param.set_value(floatX(weight))            

    def output(self, pre_act=False, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        is_tensor3_softmax = X.ndim > 2 and self.activation_str == 'softmax'

        if is_tensor3_softmax: #reshape for tensor3 softmax
            shape = X.shape
            X = X.reshape((shape[0]*shape[1], self.n_in))

        out =  self.activation(T.dot(X, self.w) + self.b)

        if is_tensor3_softmax: #reshape for tensor3 softmax
            out = out.reshape((shape[0], shape[1], self.size))

        return out

