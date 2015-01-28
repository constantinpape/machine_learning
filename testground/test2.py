import numpy as np

if __name__ == '__main__':
	A = np.ones((12,10))
	
	A[:,2] = 0.3784

	np.savetxt("test",A,fmt='%6f')
	
	B = np.genfromtxt("test")

	print B
