import numpy as np
import pylab as plot

def makeA(N, alphas):
	# determine K
	K = np.ceil(np.sqrt(2)*N) + 1
	# if K is even take K + 1
	if K % 2 == 0:
		K += 1
	# iterate over the angles
	A_matrices = []
	for alpha in alphas:
		A = np.zeros((K,N**2))
		# the positions of the sensors are given by:
		# pos_sensor_x = N/2 + k * np.cos(alpha)
		# pos_sensor_y = N/2 - k * np.sin(alpha)
		# iterate over the image positions
		for i in range(N):		# x variable
			for j in range(N):	# y variable
				# caclulate the intersection with the sensor
				k = 0.5 * ( np.sin(alpha) * (N + 1 - 2*j) - np.cos(alpha) * (N + 1 - 2*i) )
				# shift due to negative k's
				k += np.ceil(K/2)
				assert(k > 0)
				assert(k < K)
				# get the index of k
				index_k = int(k)
				# get the intensity distribution 
				intensity_k = 1 - (k - int(k))
				intensity_k1 = k - int(k)
				index = i*N + j
				assert(index < N**2)
				A[index_k, index] = intensity_k
				A[index_k+1, index] = intensity_k1
		A_matrices.append(A)
	
	returnval = np.concatenate([x for x in A_matrices])
	return returnval
				

if __name__ == '__main__':
	alphas = np.array([-33.,1.,42.])*np.pi/180.
	A = makeA(10,alphas)
	print A
