import pylab as plot
import numpy as np

methods = ( "gradient_descent", "stochastic_gradient_descent", 
	"sg_minibatch", "sg_momentum", "average_stochastic_gradient", 
	"stochastic_average_gradient", "dual_coordinate_ascent",
	"weighted_least_squares"  )

def plot_errors(data, fname): 

	fig = plot.figure()
	
	plot.plot( data[:,0], data[:,1], label = 'train_error' )
	plot.plot( data[:,0], data[:,2], label = 'test_error' )
	plot.title(fname)
	plot.xlabel('T')
	plot.ylabel('error')
	plot.legend()
	
	fname = "results/" + fname + '.png'

	fig.savefig(fname)

def plot_processing(data1, data2, fname, name1, name2):

	fig = plot.figure()
	
	plot.plot( data1[:,0], data1[:,1], label = name1 )
	plot.plot( data2[:,0], data2[:,1], label = name2 )
	plot.title("Signal Processing")
	plot.xlabel('T')
	plot.ylabel('train_error')
	plot.legend()
	
	fname = "results/" + fname + '.png'

	fig.savefig(fname)
	
if __name__ == '__main__':
	
	#for name in methods:
	#	fname = "results/" + name + ".txt" 
	#	data = np.loadtxt(fname)
	#	plot_errors(data, name)

	data_0_minibatch = np.loadtxt("/home/consti/Work/machine_learning/ml2/ex1/results/sp_minibatch.txt")
	data_0_momentum  = np.loadtxt("/home/consti/Work/machine_learning/ml2/ex1/results/sp_momentum0.txt")
	
	plot_processing(data_0_minibatch,
			data_0_momentum,
			"signal_processing0",
			"sg_momentum",
			"sg_minibatch")

	data_1_sgd	= np.loadtxt("/home/consti/Work/machine_learning/ml2/ex1/results/sp_sg.txt")
	data_1_momentum = np.loadtxt("/home/consti/Work/machine_learning/ml2/ex1/results/sp_momentum1.txt")
	
	plot_processing(data_1_sgd,
			data_1_momentum,
			"signal_processing1",
			"sg_momentum",
			"sgd")
	
