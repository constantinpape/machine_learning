import matplotlib.pyplot as plot
import matplotlib.cm as cm
import numpy as np

def plot_trainerror(data1, data2):
    fig = plot.figure()

    plot.plot( data1[0:10,0], data1[0:10,1], label = 'simple' )
    plot.plot( data2[0:10,0], data2[0:10,1], label = 'dropout' )
    plot.title('test_error')
    plot.xlabel('iteration')
    plot.ylabel('error')
    plot.legend()
    plot.show()

def plot_filter(filt):
    fig = plot.figure()

    plot.imshow( filt, interpolation = 'nearest', cmap = cm.Greys_r)
    plot.title('filter')
    plot.legend()
    plot.show()


#	fname = "results/" + fname + '.png'

#	fig.savefig(fname)

if __name__ == '__main__':
    data_nn = np.loadtxt('res/res_simple_nn')
    data_dropout = np.loadtxt('res/res_dropout_nn')
    data_conv = np.loadtxt('res/res_conv_nn')

    filter0 = np.load('res/filter/filter_0.npy')

    #plot_trainerror(data_nn,data_conv)
    plot_filter(filter0)
