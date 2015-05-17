import numpy as np
import pylab as plot

# class implementing a Naive Bayes Classifier
class DensityTreeClassifier:
	
	def __init__(self, wanted, split = True):
		self.wanted 		= wanted
		self.trained 		= False
		self.split_optimal	= split
		self.num_classes 	= []
		self.num_dimensions = []
		self.trees 			= []

	# tree data structure
	class node:

		def __init__(self,data):
			self.data 	  		= data
			self.children 		= []
			self.probability 	= []
			self.terminal		= False
			self.volume			= []
			self.depth			= []
			self.split_dim		= []
			self.split_thresh	= []
			self.calculate_volume()

		# adds a child for intermediate nodes
		def add_child(self, obj):
			if len(self.children) <= 1:
				self.children.append(obj)
			else:
				raise RuntimeError("Trying to add more than two nodes in a binary tree!")

		def calculate_volume(self):
			if self.data.shape[0] == 0:
				self.volume = 0.
			else:
				vol = 1.
				for d in range(self.data.shape[1]):
					vol *= ( max(self.data[:,d]) - min(self.data[:,d]) )
				self.volume = vol


	# check whether we have a terminal node
	# we decided for a minimal instances criterion
	def terminate_num(self, node, N_class):
		N_node = node.data.shape[0]
		N_min  = N_class**(1./3.)
		print "terminate: N", N_node, "N_min", N_min
		if N_node >= N_min:
			return False
		else:
			return True
		
	# check whether we have a terminal node
	# maximum depth criterion
	def terminate_depth(self, node):
		depth_max = 5
		# terminate for empty nodes
		if node.data.shape[0] == 0:
			return True
		# else check the criterion
		if node.depth >= depth_max:
			return True
		else:
			return False


	# calculates the probabilty for terminal nodes
	def calculate_probability(self, node, N_class):
		N_node = node.data.shape[0]
		V_node = node.volume
		if V_node == 0. and N_node == 0:
			node.probability = 0.
		elif V_node == 0 and N_node != 0:	#this may occur for N_node = 1
			node.probability = 0.
		elif V_node != 0. and N_node == 0:
			raise RuntimeError("Node Volume is not zero for empty node!")
		else:
			node.probability = N_node / (N_class * V_node) 
			
	
	# split the node according to the optimal scheme
	def split_node_optimal(self, node):
		
		# calculate the loss
		def calc_gain(node, t, N, d):
			N_l = float(np.where(node.data[:,d] < t)[0].shape[0])
			N_r = float(np.where(node.data[:,d] > t)[0].shape[0])
			
			# we don t want splits w/ 0 or 1 datapoints, because this leads to diverging gains
			if N_l <= 1 or N_r <= 1:
				return 0.
			
			else:
				#calculate the volumina
				V = node.volume
				V /= ( max(node.data[:,d]) - min(node.data[:,d]) )
				V_l = V * ( t - np.min(node.data[:,d]) )
				V_r = V * ( np.max(node.data[:,d]) - t )

				return ( N_l / N )**2 / V_l + ( N_r / N )**2 / V_r

		eps 	= 0.01
		N_node 	= node.data.shape[0]
		threshs = []
		gain 	= []
		
		# iterate over all feature dimensions to look for the best split
		for d in range(self.num_dimensions):
			threshs_d 	= []
			gain_d 		= []
			# sort the data in dimension d
			data = node.data[node.data[:,d].argsort()]
			for i in range(N_node):
				if i != 0:	# dont look left of leftmost instance
					t = data[i,d] - eps
					gain_d.append( calc_gain(node,t,N_node,d) )
					threshs_d.append( t )
				if i != N_node:	# dont look right of rightmost instance
					t = data[i,d] + eps
					gain_d.append( calc_gain(node,t,N_node,d) )
					threshs_d.append( t )
			
			gain_d = np.array( gain_d )
			# look for the best split in this dimension
			i_opt = np.argmax(gain_d)
			gain.append(gain_d[i_opt])
			threshs.append(threshs_d[i_opt])
		
		# look for the overall best split
		# optimal dimension
		d_opt = np.argmax(gain)
		# optimal threshold in the dimension
		t_opt = threshs[d_opt]
		
		# store dimension and threshold in the node
		node.split_dim 		= d_opt
		node.split_thresh 	= t_opt
		
		# split the data accordingly
		mask_l = np.where(node.data[:,d_opt] < t_opt)
		mask_r = np.where(node.data[:,d_opt] > t_opt)
		
		N_l    = mask_l[0].shape[0]
		N_r    = mask_r[0].shape[0]
		print "Splitted", N_node, "points:  Points to the left", N_l, "points to the right:", N_r
		assert(N_l + N_r == N_node)

		data_l = node.data[mask_l]
		data_r = node.data[mask_r]

		return self.node(data_l), self.node(data_r)


	# TODO How does this fit in with my tree search?
	# split the node according to the simple scheme
	def split_node_simple(self, node):
		data_l, data_r = np.array_split(node.data, 2)
		return self.node(data_l), self.node(data_r)
		

	# train the classifier
	# if train_to_generate[0] == True, only the class train_to_generate[1] is trained
	def train(self, data, labels, train_to_generate = (False,0) ):
		
		if self.trained:
			"Retraining DensityTreeClassifier!"
		
		self.trees = []
		self.priors = []

		self.num_dimensions = data.shape[1]
		print "Dimensions = ", self.num_dimensions 

		self.num_classes = len(self.wanted)
		N_samples = data.shape[0]
		# build the density tree for each class seperately
		for c in range(self.num_classes):
			
			# get all data belonging to this class
			clas = self.wanted[c]
			
			if train_to_generate[0] and clas != train_to_generate[1]:
				continue
			
			# FIXME strangest bug ever: code only runs with this print... WTF...
			print labels == clas
			mask 		= np.where(labels == clas)
			data_class 	= data[mask]
			N_class 	= data_class.shape[0]
			assert(N_class > 0)
			
			# stack
			stack = []
			# root node
			root = self.node(data_class)
			root.depth = 1
			root.split_dim = 0
			root.split_thresh = 0.
			stack.append(root)
		
			count = 0
			# build the tree
			while stack:
				# pop the last node from the stack
				curr_node = stack.pop()
				
				# check if it is a terminal node
				#if self.terminate_num(curr_node, N_class):
				if self.terminate_depth(curr_node):
					# calculate the probability
					self.calculate_probability(curr_node, N_class)
					curr_node.terminal = True
				else:
					# split the node, assign child nodes and put child nodes on the stack
					if self.split_optimal:
						# use the optimal split
						child_left, child_right = self.split_node_optimal(curr_node)
					else:
						child_left, child_right = self.split_node_simple(curr_node)
					child_left.depth = curr_node.depth + 1
					child_right.depth = curr_node.depth + 1
					curr_node.add_child(child_left)
					curr_node.add_child(child_right)
					stack.append(child_left)
					stack.append(child_right)
				count += 1
				print count

			self.trees.append(root)
		
			# calculate the prior
			self.priors.append( N_class / float(N_samples) )

		# set trained flag
		self.trained = True


	# plots the feature space splits
	def plot_splits(self):
		
		if not self.trained:
			raise RuntimeError("DensityTreeClassifier: plot_splits called, before calling train!")
		if self.num_dimensions != 2:
			raise RuntimeError("DensityTreeClassifier: plot_splits called for dimension != 2")

		curr_node = self.trees[clas]
			
		
	# walk the tree to find the correct probability for the datapoint
	def get_tree_probability(self, clas, data_point):
		# get the root node of the tree belonging to this class
		curr_node = self.trees[clas]
		
		while curr_node.terminal == False:
			dim = curr_node.split_dim
			thresh = curr_node.split_thresh
			# look whether this data_point is left or right of the split boundary
			if data_point[dim] < thresh:
				curr_node = curr_node.children[0]	
			else:
				curr_node = curr_node.children[1]
		
		prob = curr_node.probability
		return prob


	# test the classifier on test_data
	def classify(self, test_data):
		if not self.trained:
			raise RuntimeError("DensityTreeClassifier: classify called, before calling train!")
		
		results = []
		for i in range(test_data.shape[0]):
			probs = np.zeros( (self.num_classes) )
			for c in range(self.num_classes):
				data_point = test_data[i]
				# find the phase space bin in the tree and assign its probabilty
				tree_probability 	= self.get_tree_probability(c, data_point)
				probs[c] 			= self.priors[c] * tree_probability
			
			max_ind = np.argmax(probs)
			results.append(self.wanted[max_ind])

		return np.array(results)


	# generate N instances of class C
	def generate(self, N, C):
		if not self.trained:
			raise RuntimeError("DensityTreeClassifier: generate called, before calling train!")
		
		np.random.seed()
		
		C = np.searchsorted(self.wanted,C)

		generated = []
		for i in range(N):
			
			curr_node = self.trees[C]
			
			while curr_node.terminal == False:
			
				N 		= float(curr_node.data.shape[0])
				l_node 	= curr_node.children[0]
				r_node 	= curr_node.children[1]
				
				#calculate p_left
				N_l 	= float(l_node.data.shape[0])
				V_l		= l_node.volume
				p_l 	= N_l / (N * V_l) 
				
				#calculate p_right
				N_r 	= float(r_node.data.shape[0])
				V_r		= r_node.volume
				p_r 	= N_r / (N * V_r) 

				# calculate p and q
				p = p_l / (p_l + p_r)
				q = p_r / (p_l + p_r)

				rand = np.random.uniform(0.0, 1.0)

				if p < rand:
					curr_node = l_node
				else:
					curr_node = r_node

			sample_data = curr_node.data
			x = np.zeros(self.num_dimensions)
			# sample from the bin chosen
			for d in range(self.num_dimensions):
				bin_max = np.max(sample_data[:,d])
				bin_min = np.min(sample_data[:,d])
				x[d] = np.random.uniform(bin_min,bin_max)
			
			generated.append(x)
		
		return generated

