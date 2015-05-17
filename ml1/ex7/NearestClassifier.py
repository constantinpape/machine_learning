import numpy as np

class NearestClassifier:

	# @ k_params Tuple with the parameters for the k-Nearest Neighboer Classifier:
	# @ k_params[0]: Boolean specifying if k-nearest Neighbor or nearest neighbor shall be used
	# @ k_params[1]: k parameter for the k-nearest neighbor classifier
	def __init__(self, k_params = (False,1) ):
		self.enable_k_near 	= k_params[0]
		self.k 				= k_params[1]
		self.train_data 	= []
		self.train_labels 	= []
		self.test_data		= []
		self.dist_matrix	= []
		self.trained		= False
	#
	# Auxiliaty functions 
	#
	
	# calculate the euclidean distance matrix between train and test via vectorization
	def calc_dist_matrix(self):
		# tile the test data to the size of the training data
		test_tiled = np.tile(self.test_data, (self.train_data.shape[0],1, 1))
		# tile the training data to the size of the test data
		train_tiled = np.tile(self.train_data, (self.test_data.shape[0],1, 1))
		#exchange first tho dimensons of training data to adjust to test data
		train_tiled = train_tiled.transpose(1,0,2)
		# substract and square
		subst2 = np.square(train_tiled - test_tiled)
		#sum and take sqrt
		self.dist_matrix = np.sqrt( np.sum(subst2, axis = 2) )
	
	#
	# Interface
	#

	# to train the nearest neighbor classifier we only need to save the data and the labels
	def train(self, train_data, train_labels) :
		if self.trained:
			print "Retraining NearestClassifier with new data!"
		self.train_data 	= train_data
		self.train_labels 	= train_labels
		self.trained 		= True


	# classify the data with the nearest neighbor or k-nearest neighbor classifier
	def classify(self, test_data):
		if not self.trained:
			raise RuntimeError("NearestClassifier: classify called, before calling train!")
		
		self.test_data 	= test_data
		test_len  		= self.test_data.shape[0]
		train_len 		= self.train_data.shape[0]
		# calculate the euclidean distance matrix
		self.calc_dist_matrix()
		
		test_results 	=  np.zeros(test_len)
		
		# use k-nearest neighbor
		if self.enable_k_near:
			for i in range(test_len):
				dist_col = self.dist_matrix[:,i]
				assert(len(dist_col) == train_len )
				vote = []
				train_labels_copy = self.train_labels
				#find the k-nearest neighbors
				for _ in range(self.k):
					#find the current nearest neighbor
					min_dist_index = np.argmin(dist_col)
					vote.append(self.train_labels[min_dist_index])
					#delete it from the distance matrix column and the training labels
					dist_col = np.delete(dist_col, min_dist_index)
					train_labels_copy = np.delete(train_labels_copy, min_dist_index)

				test_results[i] = np.bincount(vote).argmax()
		
		# use nearest neighbor
		else:
			for i in range(test_len):
				dist_col = self.dist_matrix[:,i]
				assert(len(dist_col) == train_len )
				#find the nearest neighbor
				min_dist_index = np.argmin(dist_col)
				test_results[i] = self.train_labels[min_dist_index]
		
		return test_results
