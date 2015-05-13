import matplotlib.pyplot as plot
import matplotlib.cm as cm
import numpy as np
from load import mnist
from scipy.signal import convolve

def plot_trainerror(data1, data2, data3, data4):
    fig = plot.figure()

    plot.plot( data1[:,0], data1[:,1], label = 'simple' )
    plot.plot( data2[:,0], data2[:,1], label = 'dropout' )
    plot.plot( data3[:,0], data3[:,1], label = 'prelu' )
    plot.plot( data4[:,0], data4[:,1], label = 'conv' )
    plot.title('test_error')
    plot.xlabel('iteration')
    plot.ylabel('error')
    plot.legend(loc = 'lower right')
    plot.show()

def plot_single(data):
    fig = plot.figure()

    plot.plot( data[:,0], data[:,1], label = 'conv_aniso' )
    plot.title('anisotropic convolutional network')
    plot.xlabel('iteration')
    plot.ylabel('error')
    plot.legend(loc = 'lower right')
    plot.show()

def plot_greyscale(filt,name):
    fig = plot.figure()

    plot.imshow( filt, interpolation = 'nearest', cmap = cm.Greys_r)
    plot.title(name)
    plot.legend()
    plot.show()


if __name__ == '__main__':
    data_nn = np.loadtxt('res/res_simple_nn')
    data_dropout = np.loadtxt('res/res_dropout_nn')
    data_prelu   = np.loadtxt('res/res_prelu_nn')
    data_conv = np.loadtxt('res/res_conv_nn')
    data_aniso = np.loadtxt('res/res_conv_aniso_nn')

    # isotropic_filter
    #filter0 = np.load('res/filter/filter_0.npy')
    #filter1 = np.load('res/filter/filter_15.npy')
    #filter2 = np.load('res/filter/filter_30.npy')

    # anisotropic filter
    filter0 = np.load('res/filters_aniso/filter_0.npy')
    filter1 = np.load('res/filters_aniso/filter_15.npy')
    filter2 = np.load('res/filters_aniso/filter_30.npy')

    trX, teX, trY, teY = mnist(ntrain = 10, ntest = 10)

    #plot_trainerror(data_nn,data_dropout,data_prelu,data_conv)
    #plot_single(data_aniso)
    plot_greyscale(filter0, 'filter0')
    im = trX[5].reshape( (28,28) )
    conv = convolve(im, filter0)
    plot_greyscale(conv, 'convolution')
    plot_greyscale(filter1, 'filter15')
    conv = convolve(im, filter1)
    plot_greyscale(conv, 'convolution')
    plot_greyscale(filter2, 'filter30')
    conv = convolve(im, filter2)
    plot_greyscale(conv, 'convolution')
