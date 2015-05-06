import numpy as np
import pylab as plot
import argparse

from eval import evaluate_result

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Pass filepaths to data, target, result to evaluate classifier.')
	parser.add_argument('filepaths', metavar = 'string', type = str, nargs = 3, help = 'filepath to data, target and result')
	args = parser.parse_args().filepaths
	data 	= np.genfromtxt(args[0])
	target	= np.genfromtxt(args[1])	
	# magic magic numbers...
	for depth in (4,8,10,20):
		res_path = args[2] + "_results_" + str(depth)
		result 	= np.genfromtxt(res_path)
		evaluate_result(data, target, result)
		gen_path = args[2] + "_generated_" + str(depth)
		data_generated = np.genfromtxt(gen_path)
		for dat in data_generated:
			dat = dat.reshape( (np.sqrt(dat.shape[0]), np.sqrt(dat.shape[0])) ) 
			plot.figure()
			plot.gray()
			plot.imshow(dat, interpolation = "nearest")
			plot.show()
			plot.close()
