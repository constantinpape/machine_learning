import numpy as np

# import cross validation
from sklearn import cross_validation

# import the classifiers
from BayesClassifier 	import BayesClassifier
from QDAClassifier   	import QDAClassifier
from LDAClassifier   	import LDAClassifier
from NearestClassifier	import NearestClassifier


# returns the dataset
def read_in():
	from sklearn.datasets import load_digits
	
	digits = load_digits()

	print "Digits successfully loaded, key values:"
	print digits.keys()

	data	= digits["data"]
	labels	= digits["target"]

	# we are only interested in 3s and eights so we filter for them
	mask_3 = np.where(labels == 3)
	mask_8 = np.where(labels == 8)

	data_3 = data[mask_3]
	data_8 = data[mask_8]
	data   = np.concatenate([data_3,data_8])

	labels_3 = labels[mask_3]
	labels_8 = labels[mask_8]
	labels   = np.concatenate([labels_3,labels_8])

	# set the 3 label to class 0, the 8 label to class 1
	labels[np.where(labels == 3)] = 0
	labels[np.where(labels == 8)] = 1
	
	return data, labels


# do repeated (10 times) 2-fold cross-validation on the whole data set
# this way the reference error is determined.
def get_reference_error(data, labels, classifier):
	N = 10
	errors = np.zeros(N)
	for i in range(N):
		# do 50:50 split
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(
			data,
			labels, 
			test_size = 0.5,
			random_state = 42)
		
		#train and classify 1st time
		classifier.train(x_train,y_train)
		res = classifier.classify(x_test)
		# determine the error
		assert( res.shape[0] == y_test.shape[0] )
		false = float( len( np.where(res != y_test)[0] ) )
		err_1 = false / float( y_test.shape[0] )

		#train and classify 2nd time
		classifier.train(x_test,y_test)
		res = classifier.classify(x_train)
		# determine the error
		assert( res.shape[0] == y_train.shape[0] )
		false = float( len( np.where(res != y_train)[0] ) )
		err_2 = false / float( y_train.shape[0] )
		
		errors[i] = ( err_1 + err_2 ) / 2.
	
	return np.mean(errors)
	

if __name__ == '__main__':
	data, labels = read_in()
	
	bayes_class 	= BayesClassifier()
	qda_class		= QDAClassifier()
	lda_class		= LDAClassifier()
	nearest_class	= NearestClassifier()

	# build a dictionary
	classes 	= {'Bayes' : bayes_class, 'QDA' : qda_class, 'LDA' : lda_class, 'Nearest' : nearest_class}
	ref_errors	= {'Bayes' : 0.0, 'QDA' : 0.0, 'LDA' : 0.0, 'Nearest' : 0.0}
	
	for key in classes:
		classifier = classes[key]
		ref_errors[key] = get_reference_error(data, labels, classifier)
	
	print ref_errors
