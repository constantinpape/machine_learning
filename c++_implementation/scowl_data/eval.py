#python3.4.2
import numpy as np
#import pylab as plt
import argparse

# evaluation of the results
def evaluate_result(target, result):
    assert target.shape[0] == result.shape[0]
    
    correct = np.where( result == target )
    miss    = np.where( result != target )
    
    class_rate = float(correct[0].shape[0]) / target.shape[0]
    print("Correct classification rate:", class_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Pass filepaths to target and result for evaluation.")
    parser.add_argument("filepaths", metavar = "string", type = str,
                        nargs = 2, help = "filepath to target and result")
    args = parser.parse_args().filepaths
    test_label  = np.genfromtxt(args[0])
    result      = np.genfromtxt(args[1])
    evaluate_result(test_label, result)