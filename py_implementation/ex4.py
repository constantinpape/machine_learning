import numpy as np
import pylab as plot

import h5py

from BayesClassifier import BayesClassifier

from DensityTreeClassifier import DensityTreeClassifier

# auxiliary functions

# importing the adapted mnist data
def import_data():
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


# plotting function
def plot_diff(im1, im2):
	fig,ax = plot.subplots()
	image = ax.imshow(np.abs(im1-im2), interpolation = 'nearest')
	fig.colorbar(image)
	plot.show()
	plot.close()


# function to help for deciding which feature to use
# plots the accumaleted images of all instances belonging to the class label
def get_accumalated_image(images, target, label):
	mask = np.where(target == label)
	images_label = images[mask]
	image_accumalated = np.sum(images_label, axis = 0)
	return image_accumalated


# function for the dim-reduction 
# use the two hottest pixels for now
def reduce_dimension(data):
	len_data = data.shape[0]
	data_return = np.zeros( (len_data,2) )
	# pixel (6,4) looks most important in the diff plot
	# pixel (6,7) looks second most important
	for i in range(len_data):
		data_return[i][0] = data[i][6,4]
		data_return[i][1] = data[i][6,7]
	return data_return
	

# evaluation of the results
# scatterplot classification result
def evaluate_result(data, target, result):
	assert(data.shape[0] == target.shape[0])
	assert(target.shape[0] == result.shape[0])
	
	correct = np.where( result == target )
	miss 	= np.where( result != target )
	
	class_rate = float(correct[0].shape[0]) / target.shape[0]

	print "Correct classification rate:", class_rate 
	#get the 3s
	mask 			= np.where(target == wanted[0])
	data_3_correct 	= data[np.intersect1d(mask[0],correct[0])]
	data_3_miss	 	= data[np.intersect1d(mask[0],miss[0])]
	#get the 8s
	mask = np.where(target == wanted[1])
	data_8_correct 	= data[np.intersect1d(mask[0],correct[0])]
	data_8_miss	 	= data[np.intersect1d(mask[0],miss[0])]
	#plot
	plot.title("Scatter")
	plot.xlabel("x_0")
	plot.ylabel("x_1")
	size = 20
	plot.scatter(data_3_correct[:,0], data_3_correct[:,1], marker = "x", c = "r", s = size )
	plot.scatter(   data_3_miss[:,0],    data_3_miss[:,1], marker = "x", c = "b", s = size )
	plot.scatter(data_8_correct[:,0], data_8_correct[:,1], marker = "o", c = "r", s = size )
	plot.scatter(   data_8_miss[:,0],    data_8_miss[:,1], marker = "o", c = "b", s = size )
	plot.show()


# get the copula data
def get_copula(data):
	N 			= data.shape[0]
	dims		= data.shape[1]
	data_copula = np.zeros( (N,dims) )
	for j in range(dims):
		data_j 		= data[:,j]
		arguments_sorted = np.argsort(data_j) + 1
		data_copula[:,j] = arguments_sorted / float(N + 1)
	return data_copula


# combined generating method
def generate_threes_combined(bayes_classifier, tree_classifier, N):
	generated = []
	us = tree_classifier.generate(N,3)
	for u in us:
		x = bayes_classifier.inverse_CDF(u, 3)
		generated.append(x)
	return generated


# define the classes of interest globally
wanted = (3,8)
if __name__ == '__main__':

	# read in the data
	images_train, labels_train, images_test, labels_test = import_data()
	
	# filter data for 3s and 8s
	images_train, labels_train = filter_labels(images_train, labels_train, wanted)
	print images_train.shape[0], "train-instances of 3s and 8s in the dataset were read in."
	images_test, labels_test = filter_labels(images_test, labels_test, wanted)
	print images_test.shape[0], "test-instances of 3s and 8s in the dataset were read in."

	# accumalate and plot difference
	
	# get the accumaleted images of both classes
	images = np.concatenate([images_train,images_test])
	labels = np.concatenate([labels_train, labels_test])
	three_acc = get_accumalated_image(images, labels, wanted[0])
	eight_acc = get_accumalated_image(images, labels , wanted[1])
	#plot the difference
	plot_diff(three_acc, eight_acc)

	# reduce the dimension
	images_train_red = reduce_dimension(images_train)
	images_test_red  = reduce_dimension(images_test)
	images_train_1d = images_train.reshape( (images_train.shape[0], images_train[1]*images_train[2]) )

	naive = True
	if naive:
		# Naive Bayes Classifier
		naive_classifier = BayesClassifier(wanted)
		naive_classifier.train(images_train_red, labels_train)	
		result_bayes = naive_classifier.classify(images_test_red)
		#evaluate the result
		evaluate_result(images_test_red, labels_test, result_bayes)
		# use the classifer to generate 3s
		# first train it on the full dimensional data (needs to be 1 d)
		naive_classifier.train(images_train_1d, labels_train)	
		# then generate 10 3s
		generated_3s = naive_classifier.generate(15,3)
		generated_3s = generated_3s.reshape( (generated_3s.shape[0], np.sqrt(generated_3s.shape[1]), np.sqrt(generated_3s.shape[1]) ) )
		for three in generated_3s:
			plot.figure()
			plot.gray()
			plot.imshow(three, interpolation = "nearest")
			plot.show()
			plot.close()

	tree = False
	if tree:
		# Density Tree Classifier
		tree_classifier = DensityTreeClassifier(wanted)
		tree_classifier.train(images_train_red, labels_train)
		result_tree = tree_classifier.classify(images_test_red)
		# evaluate the result
		evaluate_result(images_test_red, labels_test, result_tree)
		# use the classifer to generate 3s
		# Need to reduce data to make this feasible
		images_train_1d = images_train_1d[0:2500]
		labels_train    = labels_train[0:2500]
		# first train it on the full dimensional data (needs to be 1 d)
		tree_classifier.train(images_train_1d, labels_train, (True,3) )
		generated_3s_tree = tree_classifier.generate(15,3)
		for three in generated_3s_tree:
			three = three.reshape( (np.sqrt(three.shape[0]), np.sqrt(three.shape[0])) ) 
			plot.figure()
			plot.gray()
			plot.imshow(three, interpolation = "nearest")
			plot.show()
			plot.close()


	combined = False
	if combined:
		#need to  use less data to make this feasible...
		images_train_1d = images_train_1d[0:2500]
		labels_train    = labels_train[0:2500]
		# combining Density Tree Classifier and Bayes Classifier
		naive_classifier_c = BayesClassifier(wanted)
		tree_classifier_c  = DensityTreeClassifier(wanted)
		# first train the naive bayes on the full feature space and the CDF
		naive_classifier_c.train(images_train_1d, labels_train)	
		naive_classifier_c.compute_CDF()
		# generate the copula data
		data_copula = get_copula(images_train_1d)
		# train the tree classifier on the copula
		tree_classifier_c.train(data_copula, labels_train, (True,3) )
		# generate the threes (10)
		generated_3s_c = generate_threes_combined(naive_classifier_c,tree_classifier_c, 50)
		for three in generated_3s_c:
			three = three.reshape( (np.sqrt(three.shape[0]), np.sqrt(three.shape[0])) ) 
			plot.figure()
			plot.gray()
			plot.imshow(three, interpolation = "nearest")
			plot.show()
			plot.close()
		


