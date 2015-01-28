import numpy as np
import pylab as plot

if __name__ == '__main__':

	data_orig = np.genfromtxt("images_train.out")
	data_c 	  = np.genfromtxt("images_train.c++")

	print np.allclose(data_orig, data_c)
	
	label_orig = np.genfromtxt("labels_train.out")
	label_c    = np.genfromtxt("labels_train.c++")

	print np.array_equal(label_orig, label_c)
