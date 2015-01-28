import numpy as np

# class implementing a Naive Bayes Classifier
class BayesClassifier:
	
	def __init__(self, wanted):
		self.wanted			= wanted
		self.num_dimensions = []
		self.num_classes	= []
		self.histos 		= []
		self.priors			= []
		self.num_bins   	= []
		self.val_ranges		= []
		self.bin_widths		= []
		self.irrelevant_dims = []
		self.trained		= False
		self.CDF			= []

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
		self.num_classes = len(self.wanted)
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
					cla = np.searchsorted(self.wanted,label)
					assert(cla < self.num_classes)
					self.histos[d][curr_bin,cla] += 1
			
		# normalize the histograms
		N_class = []
		for c in range(self.num_classes):
			clas = self.wanted[c]
			num = np.where(train_label == clas)[0].shape[0]
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
			results.append(self.wanted[clas])
		
		return np.array(results)

	
	#generate N instances of class C
	def generate(self, N, C):
		if not self.trained:
			raise RuntimeError("BayesClassifier: generate called, before calling train!")
		
		np.random.seed()
		
		C = np.searchsorted(self.wanted,C)

		generated = []
		for _ in range(N):
			gen = np.zeros(self.num_dimensions)
			# for every feature dimension sample from the corresponding histogram
			for d in range(self.num_dimensions):
				
				if d in self.irrelevant_dims:
					gen[d] = 0.
					continue
				else:
					#sampling strategy: sample from the 2 most likely histogram bins, irresponsible of their prob.
					likelihoods = self.histos[d][:,C]
					sorted_bins = np.argsort(likelihoods)
					which_bin   = np.random.randint(3)
					bin_chosen  = sorted_bins[which_bin]
					#sample the pixel value uniformly inside the chosen bin:
					bin_min = bin_chosen*self.bin_widths[d]
					bin_max = (bin_chosen + 1)*self.bin_widths[d]
					dim_val = np.random.uniform(bin_min,bin_max)
					gen[d] = dim_val
			
			generated.append(gen)
		return np.array(generated)

	
	# comute the CDF of the histograms
	def compute_CDF(self): 
		if not self.trained:
			raise RuntimeError("BayesClassifier: generate called, before calling train!")
		
		self.CDF = []
		print len(self.histos)
		print self.histos[0].shape
		
		for d in range(self.num_dimensions):
			if d in self.irrelevant_dims:
				self.CDF.append(0)	
				continue
			
			else:
				histo 	= self.histos[d]
				bins_d	= self.num_bins[d]
				assert(bins_d == histo.shape[0])
				CDF_d 	= np.zeros( (bins_d,self.num_classes) )
				for C in range(self.num_classes):
					CDF_d[0,C] = histo[0,C]
					for b in range(1,bins_d):
						CDF_d[b,C] = CDF_d[b-1,C] + histo[b,C]
				
				self.CDF.append(CDF_d)
	
	
	# compute the inverse CDF-value of x
	def inverse_CDF(self, x, clas):
		if not self.trained:
			raise RuntimeError("BayesClassifier: generate called, before calling train!")
		
		C = np.searchsorted(self.wanted, clas)
		
		assert(x.shape[0] == self.num_dimensions)

		returnval = np.zeros( (self.num_dimensions) )
		
		for d in range(self.num_dimensions):
			if d in self.irrelevant_dims:
				returnval[d] = 0.
				continue
			else:
				CDF_d = self.CDF[d][:,C]
				val = x[d]
				
				# find the bin we are in
				for j in range(CDF_d.shape[0]):
					if val < CDF_d[j]:
						break
				
				# return the position (middle of the bin)
				pos = self.bin_widths[d] * (j + 0.5)
				returnval[d] = pos
		
		return returnval
