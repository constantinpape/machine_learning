import numpy as np
import pylab as plot
import argparse

# evaluation of the results
# scatterplot classification result
# FIXME only works for 2 classes atm
def evaluate_result(data, target, result):
	assert(data.shape[0] == target.shape[0])
	assert(target.shape[0] == result.shape[0])
	
	correct = np.where( result == target )
	miss 	= np.where( result != target )
	
	class_rate = float(correct[0].shape[0]) / target.shape[0]

	print "Correct classification rate:", class_rate 
	# plot only for 2d data
	if data.shape[1] == 2:
		#get the 3s
		mask 			= np.where(target == 0)
		data_3_correct 	= data[np.intersect1d(mask[0],correct[0])]
		data_3_miss	 	= data[np.intersect1d(mask[0],miss[0])]
		#get the 8s
		mask = np.where(target == 1)
		data_8_correct 	= data[np.intersect1d(mask[0],correct[0])]
		data_8_miss	 	= data[np.intersect1d(mask[0],miss[0])]
		#plot
		plot.title("Scatter")
		plot.xlabel("x_0")
		plot.ylabel("x_1")
		size = 20
		plot.scatter(data_3_correct[:,0], data_3_correct[:,1], marker = "x", c = "r", s = size )
		plot.scatter(   data_3_miss[:,0],    data_3_miss[:,1], marker = "x", c = "b", s = size )
		plot.scatter(data_8_correct[:,0], data_8_correct[:,1], marker = "o", c = "r", s = size )
		plot.scatter(   data_8_miss[:,0],    data_8_miss[:,1], marker = "o", c = "b", s = size )
		plot.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Pass filepaths to data, target, result to evaluate classifier.')
	parser.add_argument('filepaths', metavar = 'string', type = str, nargs = 3, help = 'filepath to data, target and result')
	args = parser.parse_args().filepaths
	data 	= np.genfromtxt(args[0])
	target	= np.genfromtxt(args[1])	
	result 	= np.genfromtxt(args[2])
	evaluate_result(data, target, result)
	# look for generated data, if data has full dimension
	if data.shape[1] == 81:
		gen_path = args[2] + "_generated"
		try:
			data_generated = np.genfromtxt(gen_path)
		except:
			print "Data has full dimension, but generated data was not found."
			exit()
		for dat in data_generated:
			dat = dat.reshape( (np.sqrt(dat.shape[0]), np.sqrt(dat.shape[0])) ) 
			plot.figure()
			plot.gray()
			plot.imshow(dat, interpolation = "nearest")
			plot.show()
			plot.close()


