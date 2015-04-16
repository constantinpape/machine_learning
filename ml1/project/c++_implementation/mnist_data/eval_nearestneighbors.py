import numpy as np
import argparse

from eval import evaluate_result

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Pass filepaths to data, target, result to evaluate classifier.')
	parser.add_argument('filepaths', metavar = 'string', type = str, nargs = 3, help = 'filepath to data, target and result')
	args = parser.parse_args().filepaths
	data 	= np.genfromtxt(args[0])
	target	= np.genfromtxt(args[1])	
	# magic magic numbers...
	for k in (5,10,15,30):
		res_path = args[2] + "_results_" + str(k)
		result 	= np.genfromtxt(res_path)
		evaluate_result(data, target, result)
