import numpy as np

# class implementing a Naive Bayes Classifier
class BayesClassifier:
	
	def __init__(self):
		self.num_dimensions = []
		self.num_classes	= []
		self.histos 		= []
		self.priors			= []
		self.num_bins   	= []
		self.val_ranges		= []
		self.bin_widths		= []
		self.irrelevant_dims = []
		self.trained		= False

	# get the optimal number of bins for d-th dimension
	# using the Freedman Diaconis rule
	def get_optimal_bins(self, data, d):
		N = data.shape[0]
		# calculate the IQR
		q75, q25 = np.percentile(data[:,d], [75,25])
		iqr = q75 - q25
		# optimal width according to Freedman Diaconis
		width = 2 * iqr / N**(1./3)
		# if the width is 0, this dimension is irrelevant!
		if width == 0.0:
			self.irrelevant_dims.append(d)
			self.val_ranges.append(0.)
			self.num_bins.append(0.)
			self.bin_widths.append(0.)
			return 0
		# calculate the number of bins
		value_range = np.max(data[:,d]) - np.min(data[:,d])
		bins = int(round(value_range / width))
		
		# limit the number of bins
		if bins > N/4:
			bins = N/4
			width = value_range / bins
		
		self.val_ranges.append(value_range)
		self.num_bins.append(bins)
		self.bin_widths.append(width)
		
	
	# train the classifier by building the histograms
	def train(self, train_data, train_label): 
		if self.trained:
			print "Retraining Bayes Classifier with new data!"

		#number of classes
		self.num_classes = 2
		#number of dimensions
		self.num_dimensions = train_data.shape[1]
		
		print "Dimensions:", self.num_dimensions
		# determine the number of bins and the bin width for each dimensions individually
		self.num_bins 	= []
		self.val_ranges = []
		self.bin_widths = []
		for d in range(self.num_dimensions):
			bins = self.get_optimal_bins(train_data,d)
	
		#initialise histogram for each dimension
		self.histos = []
		for d in range(self.num_dimensions):
			self.histos.append( np.zeros( (self.num_bins[d], self.num_classes) ) )
		
		# count entries in histograms
		for i in range(train_data.shape[0]):
			for d in range(self.num_dimensions):
				if d in self.irrelevant_dims:
					continue
				else:
					val = train_data[i][d]
					# assumung that val_min = 0
					curr_bin = int( (val / self.val_ranges[d]) * self.num_bins[d] )
					# curr_bins == num_bins can happen for the maximal value
					if curr_bin == self.num_bins[d]:
						curr_bin = self.num_bins[d] - 1
					label = train_label[i]
					#get the correct class
					self.histos[d][curr_bin,label] += 1
			
		# normalize the histograms
		N_class = []
		for c in range(self.num_classes):
			num = np.where(train_label == c)[0].shape[0]
			N_class.append(num)

		for d in range(self.num_dimensions):
			for c in range(self.num_classes):
				if d in self.irrelevant_dims:
					continue
				else:
					self.histos[d][:,c] /= N_class[c]# * self.bin_widths[d])
		
		#compute the priors
		self.priors = []
		n_samples = train_label.shape[0]
		for c in range(self.num_classes):
			self.priors.append(N_class[c] / float(n_samples))
	
		self.trained = True
	

	# classify test_data
	def classify(self, test_data):
		
		if not self.trained:
			raise RuntimeError("BayesClassifier: classify called, before calling train!")

		results = []
		for i in range(test_data.shape[0]):
			probs = []
			#iterate over the classes to get the individual prob
			for c in range(self.num_classes):
				# calculate the likelihoods
				likelihood = 1.
				for d in range(self.num_dimensions):
					if d in self.irrelevant_dims:
						continue
					else:
						val = test_data[i][d]
						curr_bin = int( (val / self.val_ranges[d]) * self.num_bins[d] )
						# this may happen if test data values are bigger than train data vals
						if curr_bin >= self.num_bins[d]:
							curr_bin = self.num_bins[d] - 1
						likelihood *= self.histos[d][curr_bin][c]
				probs.append(likelihood*self.priors[c])
			
			clas = np.argmax(probs)
			results.append(clas)
		
		return np.array(results)
