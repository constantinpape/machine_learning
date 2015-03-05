from sklearn.datasets import make_classification 
import numpy as np

if __name__ == "__main__":
	#Generate a random n-class classification problem.
	data, labels    = make_classification(n_samples=5000) 
	train_data	= data[:4000,:]
	train_labels    = labels[:4000]
	test_data	= data[4000:,:]
	test_labels     = labels[4000:]

	assert train_data.shape[0] == train_labels.shape[0] == 4000
	assert test_data.shape[0] == test_labels.shape[0] == 1000 

	#save data to files
	path="original/"
	np.savetxt(path + "data_train.out",     train_data,     fmt='%f')
	np.savetxt(path + "labels_train.out",   train_labels,   fmt='%f')
	np.savetxt(path + "data_test.out",      test_data,      fmt='%f')
	np.savetxt(path + "labels_test.out",    test_labels,    fmt='%f')
	print("'.out' files save to " + path)

