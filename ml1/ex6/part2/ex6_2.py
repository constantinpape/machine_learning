import numpy as np
import matplotlib.pyplot as plot
import matplotlib.image as mpimg
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

def read_in(path):
	# read in the image
	img = mpimg.imread(path)
	# examine the image
	#print np.dtype(img)
	print img.shape
	print "max pix val", np.max(img)
	# plot the image
	plot.figure()
	plot.gray()
	plot.imshow(img[75:175,75:175])
	plot.show()
	plot.close()
	num_zeros = np.where(img==0)[0].shape[0]
	print "There are", num_zeros, "pixels with value 0 in the image."
	return img

def gaussian(x_1,x_2,sig):
	val = np.exp( - ( (x_1[0] - x_2[0])**2 + (x_1[1] - x_2[1])**2 ) / (2*sig**2) ) / ( np.sqrt(2 * np.pi) * sig )
	return val

# perform kernelized ridge regression to predict missing pixel values
# regression: y_new = y^T (K + tau * I)^-1 * kappa    w/ kappa = x_new x^T
# parameter: sig = width of gaussian, tau = ridge regression param, windowsize = window of the kernel, should be EVEN!
def ridge_regression(image, sig, tau, windowsize):
	pass
	# compute the total number of pixels
	dim_x 	= image.shape[0]
	dim_y 	= image.shape[1]
	N 		= dim_x * dim_y
	
	# compute the X-vector = (N x 2)
	# TODO use fancy meshgrid stuff
	X = np.zeros((N,2))
	for i in range(dim_x):
		for j in range(dim_y):
			index = i + j*dim_x
			X[index][0] = i
			X[index][1] = j
	
	# compute the y-vector
	Y = image.reshape( (N,1) )
	
	# compute the gaussian kernel
	# we use a gaussian kernel: K_i_j = gaussian(X_i - x_j)
	# but we only take this kernel for abs(i_1_x - i_2_x) < windowsize and abs(i_1_y -i_2_y) < windowsize
	# else the kernel is set to zero
	K = sparse.lil_matrix((N,N))
	for i_1 in range(N):
		for i in range(windowsize):
			i_2 = i_1 - (windowsize/2 - i)
			# check that we are in range (relevant at boundaries)
			if i_2 >= N or i_2 < 0:
				continue
			else:
				x_1 = X[i_1]
				x_2 = X[i_2]
				K[i_1,i_2] = gaussian(x_1,x_2,sig)
	
	K = K.tocsc()
	# make sparse
	Y_sprs = sparse.csc_matrix(Y)
	
	# precompute alpha = y^T (K + tau * I)^-1
	ridge_kernel 	= ( K + tau*sparse.identity(N) )
	inverse			= linalg.inv(ridge_kernel)

	alpha 			= Y_sprs.transpose().dot(inverse)
	print "calculated alpha:"
	print alpha.shape

	zero_pix = np.where(Y == 0)
	# predict a value for the new pixels
	# for some reason these still have to be normalized...
	new_vals 	= np.zeros( (N, 1) )
	for pix in zero_pix[0]:
		X_new = X[pix]
		X_new = X_new.reshape( (1,2) )
		kappa = X_new.dot( X.reshape( (2,N) ) )
		kappa = kappa.reshape( (N,1) )
		kappa_sprs = sparse.csc_matrix(kappa)
		y_new = alpha.dot( kappa )
		assert( y_new.shape == (1,1) )
		new_vals[pix] 	= y_new
		
	# normalise the new values
	max_val = np.max(new_vals)
	new_vals /= max_val
	Y += new_vals
	
	num_zeros = np.where(Y==0)[0].shape[0]
	print num_zeros, "zero pixel left"

	return Y.reshape( (dim_x,dim_y) )


if __name__ == '__main__':
	path 		= "cc_90.png"
	image 		= read_in(path)
	image 		= image[75:175,75:175]
	for sig in (0.5,1,2,4):
		for tau in (0.05,0.1,0.5):
			windowsize  = 6
			if sig == 4:
				windosize = 8
			image_re = ridge_regression(image, sig, tau, windowsize)
			name 	 = "regression_" + str(sig) + "_" + str(tau) 
			np.save(name, image_re)
	
