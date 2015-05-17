import matplotlib.pyplot as plot
import numpy as np

if __name__ == '__main__':
	path 		= "cc_90.png"
	for sig in (0.5,1,2,4):
		for tau in (0.05,0.1,0.5):
			name 	 = "regression_" + str(sig) + "_" + str(tau) + ".npy"
			im = np.load(name)
			
			plot.figure()
			plot.gray()
			plot.imshow(im)
			plot.show()
