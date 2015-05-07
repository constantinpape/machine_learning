import numpy as np
import time

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from load import mnist

srng = RandomStreams()

# return X as theano float array
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

# return (random) weights of given shape
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

# return rectified X
def rectify(X):
    return T.maximum(X, 0.)

# return the softmax of X
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

# optimization algorithm used
def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

# dropout: use elements of x with prob p_use,
# set to zero otherwise
def dropout(X, p_use = 1.):
        from theano.sandbox.rng_mrg import MRG_RandomStreams
        # only change X if p_use is smaller than 1
        if p_use < 1.:
            rng = MRG_RandomStreams( seed = int( time.time() ) )
            def step(x_i):
                return rng.binomial(size = x_i.shape(), n = 1, p = p_use) * x_i / p_use
            res, ups = theano.scan(step, sequences = [x])
            draw     =  theano.function(inputs = [x], outputs = [res], updates = [ups])
            X = draw(X)
        return X


# definition of 2-layer neural network
# @params:
# X : matrix of training instances
# w_h : weights input -> first hidden layer
# w_h2 : weights first hidden layer -> second hidden layer
# w_o : weights second hidden layer -> output layer
def model(X, w_h, w_h2, w_o, p_use_input, p_use_hidden):
    #X = dropout(X, p_use_input)
    # first hidden layer, activation function = relu
    h = rectify(T.dot(X, w_h))
    #h = dropout(h, p_use_hidden)
    # second hidden layer, activation function = relu
    h2 = rectify(T.dot(h, w_h2))
    #h2 = dropout(h2, p_use_hidden)
    # output layer, activation function = softmax
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

# mnist dataset, training + test
trX, teX, trY, teY = mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

# init the weights
w_h = init_weights((784, 625))
w_h2 = init_weights((625, 625))
w_o = init_weights((625, 10))

noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.8, 0.5)
h, h2, py_x = model(X, w_h, w_h2, w_o, 1., 1.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

f_out = open('res/res_simple_nn', 'w')
for i in range(1000): #you can adjust this if training takes too long
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print i
    correct = np.mean(np.argmax(teY, axis=1) == predict(teX))
    res = str(i) + " " + str(correct) + "\n"
    f_out.write( res )

