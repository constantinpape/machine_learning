import vigra 
import numpy as np
import sklearn

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


if  __name__ == '__main__':
	X, y = load_data()

	#print data.shape
	#print labels.shape
	#test_sigmoid()
	#test_gradient(X,y)
