#python3.4.2
import numpy as np
import argparse
import os.path
import math
#import pylab as plt

from mk_data import get_words

# evaluation of the results
def evaluate_result(target, result):
    assert target.shape[0] == result.shape[0]
    
    correct = np.where( result == target )
    miss    = np.where( result != target )
    
    class_rate = float(correct[0].shape[0]) / target.shape[0]
    print("Correct classification rate:", class_rate)

def iround(fnumber):
    residum = fnumber - math.floor(fnumber)
    if residum > 0.5:
        #check whether in alphabet
        assert math.ceil(fnumber) <= 25
        return math.ceil(fnumber)
    else:
        return math.floor(fnumber)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Pass filepaths to target and result for evaluation.")
    parser.add_argument("filepaths", metavar = "string", type = str,
                        nargs = 2, help = "filepath to target and result")
    args = parser.parse_args().filepaths
    
    test_label  = np.genfromtxt(args[0])
    result      = np.genfromtxt(args[1])
    evaluate_result(test_label, result)
    
    #analyze generated data
    gen_path = args[1] + "_generated"
    if os.path.isfile(gen_path):
        gen_data = np.genfromtxt(gen_path)
        words = get_words(len(gen_data[0]))
        
        print("list of generated words + true in case of existance")
        for dat in gen_data:
            word = ''.join(chr(iround(i) + ord('a')) for i in dat)
            print(word, end='')
            if word in words:
                print("\ttrue", end='')
            print()
        
    else:
        print("generated data was not found.")