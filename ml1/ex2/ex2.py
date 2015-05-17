# python source code for the first exercise in machine learning
import numpy as np
import pylab as plot

# returns the dataset
def read_in():
	from sklearn.datasets import load_digits
	digits = load_digits()
	data			= digits["data"]
	images			= digits["images"]
	target			= digits["target"]
	target_names	= digits["target_names"]
	return data, images, target, target_names


# filter data, target and images for the labels specified in labels_filter
# values in labels_filter need to be between 0 and 9
def filter_labels(data,target, images, labels_filter):
	# TODO assert that correct labels are given
	data_append = []
	target_append = []
	images_append = []
	for label in labels_filter:
		mask = np.where(target == label)
		data_label = data[mask]
		target_label = target[mask]
		images_label = images[mask]
		data_append.append(data_label)
		target_append.append(target_label)
		images_append.append(images_label)
	data_filtered = np.concatenate([x for x in data_append])
	target_filtered = np.concatenate([x for x in target_append])
	images_filtered = np.concatenate([x for x in images_append])
	return data_filtered, target_filtered, images_filtered


# split the data
def split_data(data, target):
	from sklearn import cross_validation
	x_train, x_test, y_train, y_test = cross_validation.train_test_split(
			data,
			target, 
			test_size = 0.4,	# test : train = 2 : 3 => test_size = 0.4
			random_state = 0)
	return x_train, x_test, y_train, y_test 


# function to help for deciding which feature to use
# plots the accumaleted images of all instances belonging to the class label
def get_accumalated_image(images, target, label):
	mask = np.where(target == label)
	images_label = images[mask]
	image_accumalated = np.sum(images_label, axis = 0)
	return image_accumalated


# function for the dim-reduction 
def dr(data):
	pix_per_line = 8
	len_data = data.shape[0]
	data_return = np.zeros( (len_data,2) )
	# pixel (2,3) looks most important in the diff plot
	pix_0 = 2*pix_per_line + 3
	# pixel (7,4) looks second most important
	pix_1 = 7*pix_per_line + 4
	for i in range(len_data):
		data_return[i][0] = data[i][pix_0]
		data_return[i][1] = data[i][pix_1]
	return data_return


# plotting function
def plot_diff(im1, im2):
	fig,ax = plot.subplots()
	image = ax.imshow(np.abs(im1-im2), interpolation = 'nearest')
	fig.colorbar(image)
	plot.show()
	plot.close()


#scatterplot the data
def plot_scatter(target, data):
	#get the 1s
	mask = np.where(target == 0)
	data_1 = data[mask]
	#get the sevens
	mask = np.where(target == 1)
	data_7 = data[mask]
	#plot
	plot.title("Scatter")
	plot.xlabel("x_0")
	plot.ylabel("x_1")
	size = 20
	plot.scatter(data_1[:,0], data_1[:,1], marker = "x", c = "r", s = size )
	plot.scatter(data_7[:,0], data_7[:,1], marker = "o", c = "b", s = size )
	plot.show()


# calculate the mean vectors of the classes in train_data
def get_mean_vectors(train_data, train_target):
	mean_vectors = []
	for i in range(2):
		mask = np.where(train_target == i)
		mean_vectors.append( np.mean(train_data[mask], axis = 0) )
	return np.array(mean_vectors)


# calculate the covariance matrices
def get_covariance_matrices(train_data, train_target, means):
	cov_matrices = []
	for i in range(2):
		mask = np.where(train_target == i)
		N_class = mask[0].shape[0]
		data_class = train_data[mask]
		assert(N_class == data_class.shape[0])
		# tile the means vector
		means_tiled = np.tile( means[i], (N_class,1) )
		data_subs_mean   = data_class - means_tiled
		data_subs_mean   = data_subs_mean.reshape(N_class,1,2)
		data_subs_mean_t = data_subs_mean.transpose(0,2,1) 
		cov_mat			 = np.zeros( (2,2) )
		for j in range(N_class):
			temp = data_subs_mean_t[j].dot(data_subs_mean[j])
			assert(temp.shape == (2,2))
			cov_mat += temp
		cov_mat = cov_mat / N_class
		cov_matrices.append(cov_mat)
	return np.array(cov_matrices)


# the nearest mean classifier assigns the test_data points to the class of the nearest mean vector
# for 2 classes
def nearest_mean_classifier(train_data, train_target, test_data):
	mean_vectors = get_mean_vectors(train_data, train_target)
	test_results = []
	num_classes = mean_vectors.shape[0]
	assert(num_classes == 2)
	for i in range(test_data.shape[0]):
		curr_data_tiled = np.tile(test_data[i], (num_classes,1))
		dist = np.sqrt( np.sum( np.square(curr_data_tiled - mean_vectors), axis = 0 )  )
		index = np.argmin( dist )
		test_results.append(labels[index])
	return np.array(test_results)

	
# QDA learning algorithm for 2 classes.
# returns the mean vectors, covarinace matrices and priors
def compute_qda(x_train, y_train):
	# compute mean vectors
	mean_vectors = get_mean_vectors(x_train, y_train)
	# compute the covariance matrices
	cov_matrices = get_covariance_matrices(x_train, y_train, mean_vectors)
	# compute the priors
	N_tot = y_train.shape[0]
	N_0 = np.where(y_train == 0)[0].shape[0]
	N_1 = np.where(y_train == 1)[0].shape[0]
	p_0 = float(N_0) / N_tot
	p_1 = float(N_1) / N_tot
	return mean_vectors[0], mean_vectors[1], cov_matrices[0], cov_matrices[1], p_0, p_1


# LDA learning algorithm for 2 classes.
def compute_lda(x_train, y_train):
	N 	= float( x_train.shape[0]					)
	N_0 = float( np.where(y_train == 0)[0].shape[0] )
	N_1 = float( np.where(y_train == 1)[0].shape[0] )
	# compute mean vectors
	mean_vectors = get_mean_vectors(x_train, y_train)
	# compute the overall covariance matrix by adding the (rescaled) indivdual covariance matrices
	cov_matrices = get_covariance_matrices(x_train, y_train, mean_vectors)
	cov_mat      = (N_0 * cov_matrices[0] + N_1 * cov_matrices[1]) / N
	# compute the priors
	p_0 = N_0 / N
	p_1 = N_1 / N
	# we have to return the cov_mat twice to be compatible with the qda output
	return mean_vectors[0], mean_vectors[1], cov_mat, cov_mat, p_0, p_1


# prediction with QDA for 2 classes
def perform_qda(mu_0, mu_1, cov_0, cov_1, p_0, p_1, x_test):
	test_result = []
	# calculate biases
	b_0 = 0.5 * np.log(2 * np.pi * np.linalg.det(cov_0) ) + np.log(p_0)
	b_1 = 0.5 * np.log(2 * np.pi * np.linalg.det(cov_1) ) + np.log(p_1)
	#caclulate the inverse matrices
	inv_0 = np.linalg.inv(cov_0)
	inv_1 = np.linalg.inv(cov_1)
	for i in range(x_test.shape[0]):
		# compute the likelihood for class 0
		dminm_0 = (x_test[i] - mu_0).reshape((1,2))
		dminm_0_t = dminm_0.transpose()
		likelihood_0 = 0.5 * dminm_0.dot(inv_0).dot(dminm_0_t) + b_0
		# compute the likelihood for class 1
		dminm_1 = (x_test[i] - mu_1).reshape((1,2))
		dminm_1_t = dminm_1.transpose()
		likelihood_1 = 0.5 * dminm_1.dot(inv_1).dot(dminm_1_t) + b_1
		# take class to be the smaller one
		if likelihood_0 <= likelihood_1:
			test_result.append(0)
		else:
			test_result.append(1)
	return np.array(test_result)


# prediction with LDA for 2 classes
def perform_lda(mu_0, mu_1, cov_0, cov_1, p_0, p_1, x_test):
	assert( (cov_0 == cov_1).all() )
	cov_mat = cov_0
	test_result = []
	# compute inverse
	inv = np.linalg.inv(cov_mat)
	# reshape mean vectors
	mu_0 	= mu_0.reshape((1,2))
	mu_0_t 	= mu_0.transpose()
	mu_1 	= mu_1.reshape((1,2))
	mu_1_t 	= mu_1.transpose()
	# calculate biases
	b_0 = - 0.5 * np.log( np.linalg.det(cov_mat) ) - np.log(p_0) - 0.5 * mu_0.dot(inv).dot(mu_0_t) 
	b_1 = - 0.5 * np.log( np.linalg.det(cov_mat) ) - np.log(p_1) - 0.5 * mu_1.dot(inv).dot(mu_1_t) 
	for i in range(x_test.shape[0]):
		# reshape test data
		x = x_test[i].reshape((1,2))
		likelihood_0 = x.dot(inv).dot(mu_0_t) + b_0
		likelihood_1 = x.dot(inv).dot(mu_1_t) + b_1
		if likelihood_0 >= likelihood_1:
			test_result.append(0)
		else:
			test_result.append(1)
	return np.array(test_result)


# scatterplot classification result
def evaluate_result(data, target, result):
	assert(data.shape[0] == target.shape[0])
	assert(target.shape[0] == result.shape[0])
	
	correct = np.where( result == target )
	miss 	= np.where( result != target )
	
	class_rate = float(correct[0].shape[0]) / target.shape[0]

	print "Correct classification rate:", class_rate 
	#get the 1s
	mask 			= np.where(target == 0)
	data_1_correct 	= data[np.intersect1d(mask[0],correct[0])]
	data_1_miss	 	= data[np.intersect1d(mask[0],miss[0])]
	#get the sevens
	mask = np.where(target == 1)
	data_7_correct 	= data[np.intersect1d(mask[0],correct[0])]
	data_7_miss	 	= data[np.intersect1d(mask[0],miss[0])]
	#plot
	plot.title("Scatter")
	plot.xlabel("x_0")
	plot.ylabel("x_1")
	size = 20
	plot.scatter(data_1_correct[:,0], data_1_correct[:,1], marker = "x", c = "r", s = size )
	plot.scatter(   data_1_miss[:,0],    data_1_miss[:,1], marker = "x", c = "b", s = size )
	plot.scatter(data_7_correct[:,0], data_7_correct[:,1], marker = "o", c = "r", s = size )
	plot.scatter(   data_7_miss[:,0],    data_7_miss[:,1], marker = "o", c = "b", s = size )
	plot.show()


# test template for qda and lda classifier
def test_classifier(classifier_train, classifier_predict, data, target):
	# split the data for application of the classifier
	x_train, x_test, y_train, y_test = split_data(data, target) 

	# train the classifier
	mu_0, mu_1, cov_0, cov_1, p_0, p_1 = classifier_train(x_train, y_train)	
	
	# apply to training data
	print "Running classifier on the training data."
	res_on_train = classifier_predict(mu_0, mu_1, cov_0, cov_1, p_0, p_1, x_train)
	# scatterplot the results on training
	evaluate_result(x_train, y_train, res_on_train)

	# visualize the decision boundary
	# both features are single pixel values in the range 0 to 15
	# make a grid
	grid = np.mgrid[0:15:100j,0:15:100j]
	# TODO vectorize...
	# bringing the grid in the correct data format 
	grid_data = []
	for i in range(100):
		for j in range(100):
			grid_data.append( [ grid[0][i][i], grid[1][j][j] ] )
	grid_data = np.array(grid_data)
	res_on_grid = classifier_predict(mu_0, mu_1, cov_0, cov_1, p_0, p_1, grid_data)
	# make scatterplot to visualize the decision boundary
	plot_scatter(res_on_grid, grid_data)

	# apply classifier to test data
	print "Running classifier on the testing data."
	res_on_test = classifier_predict(mu_0, mu_1, cov_0, cov_1, p_0, p_1, x_test)
	# scatterplot the results on training
	evaluate_result(x_test, y_test, res_on_test)


if __name__ == '__main__':
	
	# read in data
	data, images, target, target_names = read_in()
	# filter data for 1s and 7s
	labels = (1,7)
	data, target, images = filter_labels(data, target, images, labels)
	print data.shape[0], "instances of 1s and 7s in the dataset were read in."
	
	# get the accumaleted images of both classes
	one_acc = get_accumalated_image(images, target, labels[0])
	sev_acc = get_accumalated_image(images, target, labels[1])
	#plot the difference
	plot_diff(one_acc, sev_acc)
	
	#dimension reduction
	data_reduced = dr(data)
	print "Redueced data to dimensions:", data_reduced.shape
	
	# change classes to 0, 1.
	mask = np.where(target == labels[0])
	target[mask] = 0
	mask = np.where(target == labels[1])
	target[mask] = 1
	
	#do the scatterplot
	plot_scatter(target, data_reduced)

	# run the nearest_mean classifier, just to make sure that it is properly implemented
	# no test of the output
	res_nmc = nearest_mean_classifier(data_reduced, target, data_reduced)

	# TODO QDA results are worse than LDA -> mistake in implementation ?
	# QDA
	print "Testing QDA-Classifier"
	test_classifier(compute_qda, perform_qda, data_reduced, target)

	# LDA
	print "Testing LDA-Classifier"
	test_classifier(compute_lda, perform_lda, data_reduced, target)

