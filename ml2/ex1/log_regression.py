import vigra 
import numpy as np
import sklearn
from sklearn import cross_validation

# load the dataset from sklearn and filter for 3s and 8s
def load_data():
	from sklearn.datasets import load_digits
	digits = load_digits()
	#print digits.keys()

	data = digits["data"]
	images = digits["images"]
	target = digits["target"]
	target_names = digits["target_names"]

	#print data.shape
	#print np.dtype(data)
	
	mask_3 = np.where( target == 3)
	mask_8 = np.where( target == 8)

	data_3 = data[mask_3]
	data_8 = data[mask_8]
	target_3 = target[mask_3]
	target_8 = target[mask_8]
	
	data_return     = np.concatenate( (data_3,data_8) )
	target_return   = np.concatenate( (target_3,target_8) )

	target_return[np.where(target_return == 3)] = 1
	target_return[np.where(target_return == 8)] = -1

	return data_return, target_return

# apply sigmoid function to array Z
def sigmoid(Z):
	one = np.ones(Z.shape)
	return np.divide(one, one + np.exp(-Z) )

# test the vectorized sigmoid function
# we expect a 5,5 matrix with 1 / (1 + e**-1) = 0.73 in all entries 
def test_sigmoid():
	test_mat = np.ones( (5,5) )
	sig 	 = sigmoid(test_mat)
	print sig.shape
	print sig

# compute the gradient of the loss via vectorization
# interpret the gradient as matrix-matrix multiplication A*B (w/o normalization)
# with A = 1 - sigmoid(y X beta)
#      B = - y X
# A = 1 x N
# B = N x D
def gradient(beta, X, y):
	print X.shape, y.shape
	assert(X.shape[0] == y.shape[0])
	assert(X.shape[1] == beta.shape[0])
	M   = y.shape[0]
	y_stacked = np.tile(y, (X.shape[1],1) ).transpose()
	
	_A  = sigmoid( np.multiply(y,X.dot(beta)) )
	A   = np.ones(_A.shape) - _A
	
	B   = - np.multiply(X,y_stacked) 
	return A.dot(B) / M
	
def test_gradient(X,y):
	beta = np.ones( X.shape[1] )
	grad = gradient(beta,X,y)
	print grad.shape
	print grad

# predict labels for new instances
def predict(beta, X):
	probabilities = sigmoid( X.dot(beta) )
	assert( X.shape[0] == probabilities.shape[0] )
	predictions   = np.ones( X.shape[0] )
	mask 	      = np.where( probabilities <= 0.5)
	predictions[mask] = -1
	return predictions

# count the number of wrongly classified samples
def zero_one_loss(y_pred, y_truth):
	miss = np.where( y_pred != y_truth)
	return miss[0].shape[0]	

# learning rate update 
def tau_t(tau_0, gamma, exp, t):
	return tau_0 / ( 1 + gamma*t)**exp

# optimization methods
# syntax for all methods:
# X = training matrix
# y = training labels
# m = number of iterations
# beta_0 = initial value for the weights
# tau_0  = initial value for the learnung rate
# gamma  = update factor for the learning rate
# mu	 = parameter for sg_momentum and average stochastic gradient

# gradient descent
def gradient_descent(X, y, m, beta_0, tau_0, gamma, mu):
	beta = beta_0
	for t in range(m):
		beta -= tau_0*gradient(beta,X,y)
	return beta

# stochastic gradient descent
def stochastic_gradient_descent(X, y, m, beta_0, tau_0, gamma, mu):
	beta = beta_0
	for t in range(m):
		# choose a random training sample (sampling with replacement)
		i = np.random.randint(0, X.shape[0])
		X_i = np.expand_dims( np.array( X[i] ), axis = 0)
		y_i = np.array( [ y[i] ] )
		beta -= tau_t(tau_0,gamma,1.,t)*gradient(beta,X_i,y_i)
	return beta

# stochastic gradient descent with mini batch
def sg_minibatch(X, y, m, beta_0, tau_0, gamma, mu):
	beta = beta_0
	# minibatch size
	B = 10
	for t in range(m):
		# choose a random mini batch (with replacement)
		batch = np.random.randint(0, X.shape[0], size = B)
		beta -= tau_t(tau_0,gamma,1.,t)*gradient(beta,X[batch],y[batch]) / B
	return beta

# stochastic gradient descent with momentum
def sg_momentum(X, y, m, beta_0, tau_0, gamma, mu):
	beta = beta_0
	g    = np.zeros(beta.shape)
	for t in range(m):
		g    =  mu*g + (1 - mu)*gradient(beta, X, y)
		beta -= tau_t(tau_0,gamma,1.,t)*g
	return beta

# average stochastic gradient
def average_stochastic_gradient(X, y, m, beta_0, tau_0, gamma, mu):

	beta = beta_0
	g    = np.zeros(beta.shape)
	for t in range(m):
		# choose a random training sample (sampling with replacement)
		i = np.random.randint(0, X.shape[0])
		X_i = np.expand_dims( np.array( X[i] ), axis = 0)
		y_i = np.array( [ y[i] ] )
		g -= tau_t(tau_0,gamma,.75,t)*gradient(g, X_i, y_i)
		beta = (1 - mu)*beta + mu*g
	return beta

# stochastic avergae gradient
def stochastic_average_gradient(X, y, m, beta_0, tau_0, gamma, mu):
	beta = beta_0
	N    = X.shape[0]
	d    = np.zeros(beta.shape)
	d_mat = np.zeros( X.shape )
	for t in range(m):
		# choose a random training sample (sampling with replacement)
		i = np.random.randint(0, X.shape[0])
		d_old = d_mat[i]
		X_i = np.expand_dims( np.array( X[i] ), axis = 0)
		y_i = np.array( [ y[i] ] )
		d_mat[i] = gradient( beta, X_i, y_i )
		d += (d_mat[i] - d_old) / N
		beta -= tau_t(tau_0, gamma, 1., t)*d 
	return beta

# dual coordinate ascent
def dual_coordinate_ascent(X, y, m, beta_0, tau_0, gamma, mu):
	alpha = np.random.uniform(size = (X.shape[0]) )
	# init beta
	beta = np.multiply(alpha,y).dot(X)
	for t in range(m):
		# choose a random training sample (sampling with replacement)
		i = np.random.randint(0, X.shape[0])
		alpha_old = alpha[i]
		alpha[i] = np.clip( alpha[i] - y[i]*X[i].dot(beta) / X[i].dot(X[i]), 0, 1)
		beta += (alpha[i] - alpha_old) * y[i] * X[i]
	return beta

# weigthed least squares
# TODO
def weighted_least_squares(X, y, m, beta_0, tau_0, gamma, mu):
	N = X.shape[0]
	beta = beta_0
	for t in range(m):
		z = X.dot(beta)
		print z.shape
		V = np.sqrt( sigmoid(z).dot( np.ones(z.shape) - sigmoid(z) ) / N )
		print V.shape
		_y = np.divide(y, sigmoid( np.multiply(y,z) ) )
		_z = (z + _y).dot(V)
		_X = X.dot(V)
		# TODO use least square solver for this!
		# beta = argmin_beta( _z - X beta)**2
	return beta
		
# check if methods are working
def test_methods(X,y):
	methods = { "gradient_descent" : gradient_descent, "stochastic_gradient_descent" : stochastic_gradient_descent, 
		"sg_minbatch" : sg_minibatch, "sg_momentum" : sg_momentum, "average_stochastic_gradient": average_stochastic_gradient, 
		"stochastic_average_gradient" : stochastic_average_gradient, "dual_coordinate_ascent" : dual_coordinate_ascent,
		"weighted_least_squares" : weighted_least_squares }

	# make test/train split
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
		X,y, test_size = 0.3, random_state = 0)
	# test all methods with first row of parameters, just to check that they are working
	tau = 0.001
	mu = 0.1
	gamma = 0.0001
	beta_0 = np.zeros( X.shape[1] )
	iterations = 5
	for key in methods:
		# weighted least sqaures not properly implemented yet!
		if key == "weighted_least_squares":
			continue
		print key
		beta_res = methods[key](X_train, y_train, iterations, beta_0, tau, gamma, mu)
		prediction = predict(beta_res,X_test)
		print "worked, loss =", zero_one_loss(prediction, y_test)


if  __name__ == '__main__':
	X, y = load_data()

	#print data.shape
	#print labels.shape
	#test_sigmoid()
	#test_gradient(X,y)
	
	test_methods(X,y)	
		

