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
    N = result.shape[0]	        # number of instances
    
    # null hypothesis for two classes 0=true (words) and 1=false (nonwords)
    # classification of a test outcome given a target condition
    true_pos  = np.where( (target == 0) & (result == 0) )[0].shape[0]
    false_pos = np.where( (target == 1) & (result == 0) )[0].shape[0] #type I error
    true_neg  = np.where( (target == 1) & (result == 1) )[0].shape[0]
    false_neg = np.where( (target == 0) & (result == 1) )[0].shape[0] #type II error
    
    # target positive / negative
    pos = true_pos + false_neg
    neg = true_neg + false_pos

    # precision - positive predictive value (PPV)
    PPV = true_pos / float(true_pos + false_pos)

    # negative predictive value (NPV)
    NPV = true_neg / float(true_neg + false_neg)
    
    # sensitivity - true positive rate (TPR) - hit rate - recall
    TPR = true_pos / float(true_pos + false_neg)

    # specificity (SPC) - true negative rate (TNR)
    TNR = true_neg / float(true_neg + false_pos)

    # accuracy (ACC)
    ACC = float(true_pos  + true_neg) / N
    
    # relative accuracy & gain
    r_ACC = (float(pos) / N)**2 + (float(neg) / N)**2 
    gain_ACC  = ACC / r_ACC

    # relative precision & gain
    r_PPV = float(pos) / N
    gain_PPV = PPV / r_PPV

    print "correct classification rate (accuracy):", ACC 
    print "gain in accuracy: ", gain_ACC
    print "gain in precision: ", gain_PPV
    print "the higher the gain, the better. unity means not better than random!"
    print "words classified as nonwords (type I error): ", false_pos
    print "low if high specificity: ", TNR
    print "low if low fall-out: ", 1 - TNR 
    print "nonwords classified as words (type II error): ", false_neg
    print "low if high sensitivity: ", TPR 


def iround(fnumber):
    #check whether in alphabet
    assert math.floor(fnumber) >= 0
    assert math.ceil(fnumber) < 26
    
    residum = fnumber - math.floor(fnumber)
    if residum > 0.5:
        return int(math.ceil(fnumber))
    else:
        return int(math.floor(fnumber))


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
        words = get_words(gen_data.shape[1])
        
        print("list of generated words + true in case of existance")
        count = 0
        for dat in gen_data:
            word = ''.join(chr(iround(i) + ord('a')) for i in dat)
            if word in words:
                print word, "\ttrue"
                count += 1
            else:
                print word
        print "number of true words generated", count
        print
        
    else:
        print("generated data was not found.")
