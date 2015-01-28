import numpy as np

if __name__ == '__main__':
	A = np.ones((10,10))
	np.savetxt("test",A)
	
	B = np.genfromtxt("test")

	print B
