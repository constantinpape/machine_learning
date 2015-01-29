import numpy as np
import pylab as plot

if __name__ == '__main__':

	data_orig = np.genfromtxt("original/images_train.out")
	data_c 	  = np.genfromtxt("test_readin/images_train.out")

	print np.allclose(data_orig, data_c)
	
	label_orig = np.genfromtxt("original/labels_train.out")
	label_c    = np.genfromtxt("test_readin/labels_train.out")

	print np.array_equal(label_orig, label_c)
