import numpy as np

class QDAClassifier:

	def __init__(self):
		self.mean_vectors 	= []
		self.cov_matrices 	= []
		self.p_0			= []
		self.p_1			= []
		self.trained = False

	
	#
	# Auxiliary
	#
	
	# calculate the mean vectors of the classes in train_data
	def calculate_mean_vectors(self, train_data, train_labels):
		mean_vecs = []
		for i in range(2):
			mask = np.where(train_labels == i)
			mean_vecs.append( np.mean(train_data[mask], axis = 0) )
		self.mean_vectors = np.array(mean_vecs)

	# calculate the covariance matrices
	def calculate_covariance_matrices(self, train_data, train_labels):
		cov_matrices = []
		for i in range(2):
			mask = np.where(train_labels == i)
			N_class = mask[0].shape[0]
			data_class = train_data[mask]
			assert(N_class == data_class.shape[0])
			# tile the means vector
			means_tiled = np.tile( self.mean_vectors[i], (N_class,1) )
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
		self.cov_matrices = np.array(cov_matrices)

	#
	# Interface
	#
	
	# train the classifier on train_data and train_labels
	def train(self, train_data, train_labels):
		if self.trained:
			print "Retraining LDAClassifier with new data!"
		
		# compute mean vectors
		self.calculate_mean_vectors(train_data, train_labels)
		# compute the covariance matrices
		self.calculate_covariance_matrices(train_data, train_labels)
		# compute the priors
		N_tot = train_labels.shape[0]
		N_0 = np.where(train_labels == 0)[0].shape[0]
		N_1 = np.where(train_labels == 1)[0].shape[0]
		self.p_0 = float(N_0) / N_tot
		self.p_1 = float(N_1) / N_tot

		self.trained = True

	# classify test_data
	# prediction with QDA for 2 classes
	def classify(self, test_data):
		if not self.trained:
			raise RuntimeError("LDAClassifier: classify called, before calling train!")
		test_result = []
		# calculate biases
		b_0 = 0.5 * np.log(2 * np.pi * np.linalg.det(self.cov_matrices[0]) ) + np.log(self.p_0)
		b_1 = 0.5 * np.log(2 * np.pi * np.linalg.det(self.cov_matrices[1]) ) + np.log(self.p_1)
		#caclulate the inverse matrices
		inv_0 = np.linalg.inv(self.cov_matrices[0])
		inv_1 = np.linalg.inv(self.cov_matrices[1])
		for i in range(test_data.shape[0]):
			# compute the likelihood for class 0
			dminm_0 = (test_data[i] - self.mean_vectors[0]).reshape((1,2))
			dminm_0_t = dminm_0.transpose()
			likelihood_0 = 0.5 * dminm_0.dot(inv_0).dot(dminm_0_t) + b_0
			# compute the likelihood for class 1
			dminm_1 = (x_test[i] - self.mean_vectors[1]).reshape((1,2))
			dminm_1_t = dminm_1.transpose()
			likelihood_1 = 0.5 * dminm_1.dot(inv_1).dot(dminm_1_t) + b_1
			# take class to be the smaller one
			if likelihood_0 <= likelihood_1:
				test_result.append(0)
			else:
				test_result.append(1)
		return np.array(test_result)
