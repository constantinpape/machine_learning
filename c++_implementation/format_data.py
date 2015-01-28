import numpy as np
import h5py

# importing the adapted mnist data
def import_data_mnist():
	test_path     = "../data/mnist/small/test.h5"
	training_path = "../data/mnist/small/train.h5"
	
	f = h5py.File(training_path, 'r')
	images_train = f["images"].value
	labels_train = f["labels"].value
	f.close()
	
	f = h5py.File(test_path, 'r')
	images_test = f["images"].value
	labels_test = f["labels"].value
	f.close()
	
	print "images_train shape:"
	print np.shape(images_train)
	print "labels_train shape:"
	print np.shape(labels_train)
	print "images_test shape:"
	print np.shape(images_test)
	print "labels_test shape:"
	print np.shape(labels_test)

	return images_train, labels_train, images_test, labels_test

# filter data, target and images for the labels specified in labels_filter
# values in labels_filter need to be between 0 and 9
def filter_labels(data, target, labels_filter):
	data_append = []
	target_append = []
	for label in labels_filter:
		mask = np.where(target == label)
		data_label = data[mask]
		target_label = target[mask]
		data_append.append(data_label)
		target_append.append(target_label)
	data_filtered = np.concatenate([x for x in data_append])
	target_filtered = np.concatenate([x for x in target_append])
	return data_filtered, target_filtered

if __name__ == '__main__':
	# read in the data
	images_train, labels_train, images_test, labels_test = import_data_mnist()
	
	# filter data for 3s and 8s
	images_train, labels_train = filter_labels(images_train, labels_train, (3,8))
	print images_train.shape[0], "train-instances of 3s and 8s in the dataset were read in."
	images_test, labels_test = filter_labels(images_test, labels_test, (3,8))
	print images_test.shape[0], "test-instances of 3s and 8s in the dataset were read in."

	# reshape the image data
	images_train = images_train.reshape(images_train.shape[0],images_train.shape[1]*images_train.shape[2])
	print "Train data reshaped to", images_train.shape
	images_test = images_test.reshape(images_test.shape[0],images_test.shape[1]*images_test.shape[2])
	print "Test data reshaped to", images_test.shape

	# change labels 3 -> 0, 8 -> 1
	labels_train[np.where(labels_train==3)] = 0
	labels_train[np.where(labels_train==8)] = 1
	
	labels_test[np.where(labels_test==3)]   = 0
	labels_test[np.where(labels_test==8)]   = 1

	# save to read in via c++
	np.savetxt("mnist_data/images_train.out", images_train, fmt='%f')
	np.savetxt("mnist_data/labels_train.out", labels_train, fmt='%i')
	np.savetxt("mnist_data/images_test.out",  images_test,  fmt='%f')
	np.savetxt("mnist_data/labels_test.out",  labels_test,  fmt='%i')
	
