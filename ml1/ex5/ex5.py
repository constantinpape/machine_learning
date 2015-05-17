import numpy as np
import pylab as plt
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

#np.set_printoptions(threshold='nan')


#NOTE: old angles and images used!
def makeA(shape, alphas):
    # determine K
    K = np.ceil(np.linalg.norm(shape))
    # if K is even take K + 1
    if K % 2 == 0:
        K += 1
    
    A = sp.dok_matrix((K*len(alphas), shape[0]*shape[1]), dtype=np.float64)
    # iterate over the angles
    for a in range(len(alphas)):
        print alphas[a]
        #NOTE: alternate definition for alpha (different from fig.2)
        #      alpha = angle(x_axis, sensor)
        # the positions of the sensors are given by:
        # pos_sensor_x = dim_x/2 + k * cos(alpha)
        # pos_sensor_y = dim_y/2 + k * sin(alpha)
        # define ray vector once
        ray = np.array([np.cos(alphas[a]*np.pi/180.), np.sin(alphas[a]*np.pi/180.)])
        
        # iterate over the image positions
        for i in range(shape[0]):		# x variable
            for j in range(shape[1]):	        # y variable
                # tesselation, origin top left corner
                ij = np.array([i - 0.5*(shape[0] - 1), j - 0.5*(shape[1] - 1 )])
                # caclulate the intersection with the sensor
                k = np.inner(ray, ij)
                
                # first sensorpixel closest to x-axis
                if alphas[a] < 0:
                    k *= -1
                # shift due to negative k's
                k += 0.5*(K-1)
                assert(k > 0)
                assert(k < K)
                # get the index of k
                index_k = a*K + int(k)
                assert(index_k < K*len(alphas))
                # get the intensity distribution
                intensity_k = 1 - (k - int(k))
                intensity_k1 = k - int(k)
                index = j*shape[0] + i
                assert(index < shape[0]*shape[1])
                
                A[index_k, index] = intensity_k
                A[index_k + 1, index] = intensity_k1
    
    return A


def reconstruct_x(A, y, lim=None):
    # convert to compressed sparse column matrix
    A = A.tocsc()
    # solve unsym LinEq via least squares
    x = lsqr(A, y, iter_lim=lim, show=True)
    return x


if __name__ == '__main__':
   
	# test the makeA function
	alphas = np.array([-33.,1.,42.])
	shape = (10,10)
	A = makeA(shape,alphas)
	A = A.todense()
	plt.gray()
	plt.imshow(A)
	plt.savefig('test_matrix.png')

	part_ii = True
	# second part: reconstruct the low-res image
	if part_ii:
		shape = (77, 77)
		y = np.load('hs_tomography/y_81_N5929_K109.npy')
		alphas = np.load('hs_tomography/y_81_N5929_K109_alphas.npy')
    	A = makeA(shape,alphas)
	
	# third part: reconstruct the high-res image
	part_iii = False
	if part_iii:
		shape = (195, 195)
		y = np.load('hs_tomography/y_201_N38025_K_276.npy')
		alphas = np.load('hs_tomography/y_201_N38025_K_276_alphas.npy')
    	A = makeA(shape,alphas)
	
	if part_ii or part_iii:
		x = reconstruct_x(A, y, lim=1e3)
		x = np.reshape(x[0], shape)
    	plt.gray()
    	plt.imshow(x)
    	plt.savefig('out.png')
