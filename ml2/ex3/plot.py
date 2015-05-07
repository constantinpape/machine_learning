import pylab as plot
import numpy as np

def plot_trainerror(data):
    fig = plot.figure()

    plot.plot( data[:,0], data[:,1], label = 'train_error' )
    #plot.title(fname)
    plot.xlabel('iteration')
    plot.ylabel('error')
    plot.legend()
    plot.show()

#	fname = "results/" + fname + '.png'

#	fig.savefig(fname)

if __name__ == '__main__':
    data = np.loadtxt('res/res_simple_nn')
    plot_trainerror(data)
