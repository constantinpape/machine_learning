#python3.4.2
import numpy as np
import argparse
from sklearn.datasets import load_iris
import os.path
import math
#import pylab as plt


# evaluation of the results
def evaluate_result(target, result):
        assert(target.shape[0] == result.shape[0])
        correct = np.where( result == target )
        miss 	= np.where( result != target )

        class_rate = float(correct[0].shape[0]) / target.shape[0]

        print "Correct classification rate:", class_rate 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Pass filepaths to target and result for evaluation.")
    parser.add_argument("filepaths", metavar = "string", type = str,
                        nargs = 2, help = "filepath to target and result")
    args = parser.parse_args().filepaths
    
    test_label  = np.genfromtxt(args[0])
    result      = np.genfromtxt(args[1])
    evaluate_result(test_label, result)
    class_names = load_iris().target_names
    
    np.set_printoptions(threshold=np.inf)
    print load_iris().data
    #analyze generated data
    gen_path = args[1] + "_generated"
    if os.path.isfile(gen_path):
        gen_data = np.genfromtxt(gen_path)
        print gen_data 
    else:
        print("generated data was not found.")
