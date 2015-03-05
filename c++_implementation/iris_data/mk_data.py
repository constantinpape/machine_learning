from sklearn.datasets import load_iris
import numpy as np

#load data and concatenate
def get_iris():
	data = load_iris()
	class_names = data.target_names

	data = np.c_[data.data, data.target]
	data = np.random.permutation(data)
	return data[:,:-1], data[:,-1]

if __name__ == "__main__":
	#split data into train and test sets
	data, labels    = get_iris()
	split_indx      = int( np.floor( 0.8*data.shape[0] ) )
	train_data      = data[:split_indx]
	train_labels    = labels[:split_indx]
	test_data       = data[split_indx:]
	test_labels     = labels[split_indx:]


	#save data to files
	path="original/"
	np.savetxt(path + "data_train.out",     train_data,     fmt='%f')
	np.savetxt(path + "labels_train.out",   train_labels,   fmt='%f')
	np.savetxt(path + "data_test.out",      test_data,      fmt='%f')
	np.savetxt(path + "labels_test.out",    test_labels,    fmt='%f')
	print("'.out' files save to " + path)

