import numpy as np
import time

#from ipython import embed

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal.downsample import max_pool_2d

from load import mnist

srng = RandomStreams()

# return X as theano float array
def floatX(X):
    print theano.config.floatX
    return np.asarray(X, dtype=theano.config.floatX)

# return (random) weights of given shape
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

# return rectified X
def rectify(X):
    return T.maximum(X, 0.)

def PRelu(X,a):
    return a* X * (X <= 0) + X* ( X > 0 )

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
    # only change X if p_use is smaller than 1
    if p_use < 1.:

        binomial = srng.binomial(size = X.shape, n = 1, p = p_use, dtype = theano.config.floatX) * X / p_use
        return binomial
    else:
        return X


# definition of 2-layer neural network
# @params:
# X : matrix of training instances
# w_h : weights input -> first hidden layer
# w_h2 : weights first hidden layer -> second hidden layer
# w_o : weights second hidden layer -> output layer
# p_use_input: probability to keep neuron in dropout, input layer
# p_use_hidden: probability to keep neuron in dropout, hidden layer
def model(
        X,
        w_h,
        w_h2,
        w_o,
        p_use_input,
        p_use_hidden
        ):
    X = dropout(X, p_use_input)
    # first hidden layer, activation function = relu
    h = rectify(T.dot(X, w_h))
    h = dropout(h, p_use_hidden)
    # second hidden layer, activation function = relu
    h2 = rectify(T.dot(h, w_h2))
    h2 = dropout(h2, p_use_hidden)
    # output layer, activation function = softmax
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

# run the simple neural network (with dropout)
def simple_nn():
    # mnist dataset, training + test
    trX, teX, trY, teY = mnist(onehot=True)

    X = T.fmatrix()
    Y = T.fmatrix()

    # init the weights
    w_h = init_weights((784, 625))
    w_h2 = init_weights((625, 625))
    w_o = init_weights((625, 10))

    noise_h, noise_h2, noise_py_x = model(
            X,
            w_h,
            w_h2,
            w_o,
            0.8,
            0.5
            )
    h, h2, py_x = model(
            X,
            w_h,
            w_h2,
            w_o,
            1.,
            1.
            )
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    params = [w_h, w_h2, w_o]
    updates = RMSprop(cost, params, lr=0.001)

    train = theano.function(
            inputs=[X, Y],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True
            )
    predict = theano.function(
            inputs=[X],
            outputs=y_x,
            allow_input_downcast=True
            )

    f_out = open('res/res_dropout_nn', 'w')
    for i in range(10): #you can adjust this if training takes too long
        # train batches of 128 instances
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            cost = train(trX[start:end], trY[start:end])
        print i
        correct = np.mean(np.argmax(teY, axis=1) == predict(teX))
        res = str(i) + " " + str(correct) + "\n"
        f_out.write( res )

# definition of 2-layer neural network with prelu activation function
# @params:
# X : matrix of training instances
# w_h : weights input -> first hidden layer
# w_h2 : weights first hidden layer -> second hidden layer
# w_o : weights second hidden layer -> output layer
# a_1 : prelu parameters for first hidden layer
# a_2 : prelu parameters for second hidden layer
# p_use_input: probability to keep neuron in dropout, input layer
# p_use_hidden: probability to keep neuron in dropout, hidden layer
def model_prelu(
        X,
        w_h,
        w_h2,
        w_o,
        a_1,
        a_2,
        p_use_input,
        p_use_hidden
        ):
    X = dropout(X, p_use_input)
    # first hidden layer, activation function = relu
    h = PRelu(T.dot(X, w_h), a_1)
    h = dropout(h, p_use_hidden)
    # second hidden layer, activation function = relu
    h2 = PRelu(T.dot(h, w_h2),a_2)
    h2 = dropout(h2, p_use_hidden)
    # output layer, activation function = softmax
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

# run the neural network with prelu activation function
def prelu_nn():
    # mnist dataset, training + test
    trX, teX, trY, teY = mnist(onehot=True)

    X = T.fmatrix()
    Y = T.fmatrix()

    # init the weights
    w_h = init_weights((784, 625))
    w_h2 = init_weights((625, 625))
    w_o = init_weights((625, 10))
    a_1 = init_weights( (625,) )
    a_2 = init_weights( (625,) )

    noise_h, noise_h2, noise_py_x = model_prelu(
            X,
            w_h,
            w_h2,
            w_o,
            a_1,
            a_2,
            0.8,
            0.5
            )
    h, h2, py_x = model_prelu(
            X,
            w_h,
            w_h2,
            w_o,
            a_1,
            a_2,
            1.,
            1.
            )

    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    params = [w_h, w_h2, w_o, a_1, a_2]
    updates = RMSprop(cost, params, lr=0.001)

    train = theano.function(
            inputs=[X, Y],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True
            )
    predict = theano.function(
            inputs=[X],
            outputs=y_x,
            allow_input_downcast=True
            )

    f_out = open('res/res_prelu_nn', 'w')

    for i in range(10): #you can adjust this if training takes too long
        # train batches of 128 instances
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            cost = train(trX[start:end], trY[start:end])
        print i
        correct = np.mean(np.argmax(teY, axis=1) == predict(teX))
        res = str(i) + " " + str(correct) + "\n"
        f_out.write( res )

# definition of convolutional neural network
# @params:
# X : matrix of training instances
# w_1 : weights input -> first layer of convolutional net
# w_2 : weights first conv layer -> second conv layer
# w_3 : weights second conv layer -> third conv layer
# w_h2 : weights third conv -> second hidden layer
# w_o : weights second hidde layer -> output layer
# p_use_input: probability to keep neuron in dropout, input layer
# p_use_hidden: probability to keep neuron in dropout, hidden layer
def model_conv(
        X,
        w_1,
        w_2,
        w_3,
        w_h2,
        w_o,
        p_use_input,
        p_use_hidden
        ):
    X = dropout(X, p_use_input)

    # first convolutional layer:
    conv_layer_1 = rectify( T.nnet.conv2d(X, w_1, border_mode = 'full' ))
    sub_layer_1  = T.signal.downsample.max_pool_2d(conv_layer_1, (2, 2) )
    out_1   = dropout(sub_layer_1, p_use_input)

    # second convolutional layer:
    conv_layer_2 = rectify( T.nnet.conv2d(out_1, w_2) )
    sub_layer_2  = T.signal.downsample.max_pool_2d(conv_layer_2, (2, 2) )
    out_2   = dropout(sub_layer_2, p_use_hidden)

    # third convolutional layer:
    conv_layer_3 = rectify( T.nnet.conv2d(out_2, w_3) )
    sub_layer_3  = T.signal.downsample.max_pool_2d(conv_layer_3, (2, 2) )
    out_3   = dropout(sub_layer_3, p_use_hidden)

    out_3 = T.flatten(out_3, outdim = 2)
    h2 = rectify(T.dot(out_3, w_h2))
    h2 = dropout(h2, p_use_hidden)
    # output layer, activation function = softmax
    py_x = softmax(T.dot(h2, w_o))
    return out_1, out_2, out_3, h2, py_x

def conv_nn():
    # mnist dataset, training + test
    trX, teX, trY, teY = mnist(onehot=True)

    trX = trX.reshape(-1, 1, 28, 28)
    teX = teX.reshape(-1, 1, 28, 28)

    X = T.ftensor4()
    Y = T.fmatrix()

    # init the weights
    w_1 = init_weights((32, 1, 5, 5))
    w_2 = init_weights((64, 32, 5, 5))
    w_3 = init_weights((128, 64, 2, 2))
    #number of pixel in last conv layer:
    #num_filter * pix_per_filter = 128 * 9 = 1152
    w_h2 = init_weights((1152, 625 ))
    w_o = init_weights((625, 10))

    noise_out_1, noise_out_2, noise_out_2, noise_h2, noise_py_x = model_conv(
            X,
            w_1,
            w_2,
            w_3,
            w_h2,
            w_o,
            0.8,
            0.5
            )
    out_1, out_2, out_3, h2, py_x = model_conv(
            X,
            w_1,
            w_2,
            w_3,
            w_h2,
            w_o,
            1.,
            1.
            )
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    params = [w_1, w_2, w_3, w_h2, w_o]
    updates = RMSprop(cost, params, lr=0.001)

    train = theano.function(
            inputs=[X, Y],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True
            )
    predict = theano.function(
            inputs=[X],
            outputs=y_x,
            allow_input_downcast=True
            )

    f_out = open('res/res_conv_nn', 'w')
    for i in range(10): #you can adjust this if training takes too long
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            cost = train(trX[start:end], trY[start:end])
        print i
        correct = np.mean(np.argmax(teY, axis=1) == predict(teX))
        res = str(i) + " " + str(correct) + "\n"
        f_out.write( res )

if __name__ == '__main__':
    #simple_nn()
    #prelu_nn()
    conv_nn()
