# python source code for the first exercise in machine learning
import numpy as np
import pylab as plot

# returns the dataset
def read_in():
	from sklearn.datasets import load_digits
	
	digits = load_digits()

	print "Digits successfully loaded, key values:"
	print digits.keys()

	data			= digits["data"]
	images			= digits["images"]
	target			= digits["target"]
	target_names	= digits["target_names"]
	
	return data, images, target, target_names

# explore the digits-data from sklearn.dataset
# corresponds to exercise 1
def explore_data(data, images, target):

	# try to determine the type of data...
	print "data_type belonging to key data:"
	try: 
		print np.dtype(data)
	except TypeError as err:
		print err
	
	print "It has dimension", np.shape(data)

	# plot a 3
	
	# get indices of all threes in target
	threes = np.where(target == 3)
	#assert threes is not empty
	assert(len(threes) > 0)
	# choose the first 3
	three_indx = threes[0]
	# get the image
	img = images[three_indx][0]

	#plot it
	plot.figure()
	plot.gray()
	plot.imshow(img, interpolation = "nearest")
	plot.show()
	plot.close()

# split the data
def split_data(data, target):
	from sklearn import cross_validation
	x_train, x_test, y_train, y_test = cross_validation.train_test_split(
			data,
			target, 
			test_size = 0.4,
			random_state = 0)
	return x_train, x_test, y_train, y_test 

# calculate the euclidean distance matrix between train and test via loops
def dist_loop(train, test):
	len_train = np.shape(train)[0]
	len_test  = np.shape(test)[0]
	euclid_dist = np.zeros((len_train,len_test))
	#iterate training data
	for i in range(len_train):
		#iterate test data
		for j in range(len_test):
			assert(len(train[i]) == len(test[j]))
			# i hope it is okay not to use a loop here...
			distances = train[i] - test[j]
			euclid_dist[i,j] = np.sqrt(np.sum(np.square(distances)))
	return euclid_dist

# calculate the euclidean distance matrix between train and test via vectorization
def dist_vec(train, test):
	# tile the test data to the size of the training data
	test_tiled = np.tile(test, (train.shape[0],1, 1))
	# tile the training data to the size of the test data
	train_tiled = np.tile(train, (test.shape[0],1, 1))
	#exchange first tho dimensons of training data to adjust to test data
	train_tiled = train_tiled.transpose(1,0,2)
	# substract and square
	subst2 = np.square(train_tiled - test_tiled)
	#sum and take sqrt
	euclid_dist = np.sqrt( np.sum(subst2, axis = 2) )
	return euclid_dist

# nearest neighbor classifier
# the argument k is not used, but it is here for convenience (to make test_classifier applicable to both classifiers)
def nn_classifier(train_data, test_data, train_labels, k = 1):
	test_len  = test_data.shape[0]
	train_len = train_data.shape[0]

	dist_matrix = dist_vec(train_data, test_data)
	test_results = np.zeros( (test_len), dtype=int)
	for i in range(test_len):
		dist_col = dist_matrix[:,i]
		assert(len(dist_col) == train_len )
		#find the nearest neighbor
		min_dist_index = np.argmin(dist_col)
		test_results[i] = train_labels[min_dist_index]
	
	return test_results

# k - nearest neighbor classifier
def knn_classifier(train_data, test_data, train_labels, k):
	test_len  = test_data.shape[0]
	train_len = train_data.shape[0]

	dist_matrix = dist_vec(train_data, test_data)
	test_results = np.zeros( (test_len), dtype=int)
	for i in range(test_len):
		dist_col = dist_matrix[:,i]
		assert(len(dist_col) == train_len )
		vote = []
		train_labels_copy = train_labels
		#find the k-nearest neighbors
		for _ in range(k):
			#find the current nearest neighbor
			min_dist_index = np.argmin(dist_col)
			vote.append(train_labels[min_dist_index])
			#delete it from the distance matrix column and the training labels
			dist_col = np.delete(dist_col, min_dist_index)
			train_labels_copy = np.delete(train_labels_copy, min_dist_index)

		test_results[i] = np.bincount(vote).argmax()
	
	return test_results

# filter data and target for the labels specified in labels_filter
# values in labels_filter need to be between 0 and 9
def filter_labels(data,target,labels_filter):
	# TODO assert that correct labels are given
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

# evaluate the result of the classifier by returning the correct classification rate
def evaluate_result(y_test, results_test, label):
	assert(len(y_test) == len(results_test))
	correct = len(np.where(results_test == y_test)[0])
	return float(correct) / len(y_test)

# test the classifier
def test_classifier(classifier, data, target, labels, k = 1):
	data_filtered, target_filtered = filter_labels(data, target, labels)
	x_train, x_test, y_train, y_test = split_data(data_filtered, target_filtered)
	results_test = classifier(x_train, x_test, y_train, k)
	return evaluate_result(y_test, results_test, labels[0])

# cross validate the nearest neighbor classifier
def cross_validation(data, target, n):
	# split the data in n parts
	# TODO handle non-integer segment_len properly
	segment_len = int(data.shape[0] / n)
	data_split = []
	target_split = []
	for i in range(n):
		i_0 = i * segment_len
		i_1 = (i + 1) * segment_len
		data_split.append(data[i_0:i_1])
		target_split.append(target[i_0:i_1])
	
	correct_rates = np.zeros((n))
	for i in range(n):
		x_test = data_split[i]
		y_test = target_split[i] 
		x_train = np.concatenate( [z for z in np.delete(  data_split, i, axis=0)] )
		y_train = np.concatenate( [z for z in np.delete(target_split, i, axis=0)] )
		result = nn_classifier(x_train, x_test, y_train)
		assert( len(result) == len(y_test) )
		correct = len(np.where(result == y_test)[0])
		correct_rates[i] = float(correct) / len(y_test)
	
	return np.mean(correct_rates), np.var(correct_rates)

if __name__ == '__main__':
	data, images, target, target_names = read_in()
	explore_data(data, images, target)
	
	x_train,x_test,y_train,y_test = split_data(data, target)
	
	import time as t

	t_0 = t.time()
	loop_dist = dist_loop(x_train,x_test)
	t_1 = t.time()
	t_loop = t_1 - t_0
	print "Calculate distance matrix with loops in", t_loop, "s"

	t_2 = t.time()
	vec_dist = dist_vec(x_train,x_test)
	t_3 = t.time()
	t_vec = t_3 - t_2
	print "Calculate distance matrix with vectorization in", t_vec, "s"

	# assert that both method yield the same results
	assert(vec_dist.shape == loop_dist.shape)
	assert( np.array_equal(vec_dist,loop_dist) )
	print "Both methods yield the same result."

	print "Nearest Neighbor Classifier:"

	labels = (1,3)
	error_rate = test_classifier(nn_classifier, data, target, labels)
	print "Classification rate for distinguishing 1 and 3:", error_rate
	
	labels = (1,7)
	error_rate = test_classifier(nn_classifier, data, target, labels)
	print "Classification rate for distinguishing 1 and 7:", error_rate

	print "k-Nearest Neighbor Classifier"
	for k in (1,3,5,9,17,33):
		error_rate = test_classifier(knn_classifier, data, target, labels)
		print "k =", k, ": Classification rate for distinguishing 1 and 7:", error_rate 

	print "Cross validation"
	for n in (2,5,10):
		mean, var = cross_validation(data, target, n)
		print "For n =", n, ": mean classifcation rate =", mean, "with variance = ", var
