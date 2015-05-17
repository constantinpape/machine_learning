import numpy as np

class LDAClassifier:

	def __init__(self):
		self.mean_vectors 	= []
		self.cov_matrices 	= []
		self.p_0			= []
		self.p_1			= []
		self.trained 		= False

	
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
			print data_subs_mean.shape, N_class*1*2
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
	
	def train(self, train_data, train_labels):
		if self.trained:
			print "Retraining LDAClassifier with new data!"

		N 	= float( train_data.shape[0]					 )
		N_0 = float( np.where(train_labels == 0)[0].shape[0] )
		N_1 = float( np.where(train_labels == 1)[0].shape[0] )
		# compute mean vectors
		self.calculate_mean_vectors(train_data, train_labels)
		# compute the overall covariance matrix by adding the (rescaled) indivdual covariance matrices
		self.calculate_covariance_matrices(train_data, train_labels)
		self.cov_matrices  = (N_0 * self.cov_matrices[0] + N_1 * self.cov_matrices[1]) / N
		# compute the priors
		self.p_0 = N_0 / N
		self.p_1 = N_1 / N

		self.trained = True

	
	def classify(self, test_data):
		if not self.trained:
			raise RuntimeError("LDAClassifier: classify called, before calling train!")
			
		test_result = []
		# compute inverse
		inv = np.linalg.inv(self.cov_matrices)
		# reshape mean vectors
		mu_0 	= self.mean_vectors[0].reshape((1,2))
		mu_0_t 	= mu_0.transpose()
		mu_1 	= self.mean_vectors[1].reshape((1,2))
		mu_1_t 	= mu_1.transpose()
		# calculate biases
		b_0 = - 0.5 * np.log( np.linalg.det(self.cov_matrices) ) - np.log(self.p_0) - 0.5 * mu_0.dot(inv).dot(mu_0_t) 
		b_1 = - 0.5 * np.log( np.linalg.det(self.cov_matrices) ) - np.log(self.p_1) - 0.5 * mu_1.dot(inv).dot(mu_1_t) 
		for i in range(test_data.shape[0]):
			# reshape test data
			x = test_data[i].reshape((1,2))
			likelihood_0 = x.dot(inv).dot(mu_0_t) + b_0
			likelihood_1 = x.dot(inv).dot(mu_1_t) + b_1
			if likelihood_0 >= likelihood_1:
				test_result.append(0)
			else:
				test_result.append(1)
		return np.array(test_result)
