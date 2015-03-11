from sklearn.datasets import make_classification 
import numpy as np
import argparse

def mk_madelon(complexity, N=5000):
    print 'generating problem of complexity:', complexity

    #Generate a random n-class classification problem.
    split = int(0.8 * N) 
    data, labels= make_classification(n_samples=N, n_informative=complexity) 
    train_data	= data[:split,:]
    train_labels= labels[:split]
    test_data	= data[split:,:]
    test_labels = labels[split:]

    assert train_data.shape[0] == train_labels.shape[0] == split 
    assert test_data.shape[0]  == test_labels.shape[0]  == N - split 

    #save data to files
    path="original/"
    np.savetxt(path + "data_train.out",     train_data,     fmt='%f')
    np.savetxt(path + "labels_train.out",   train_labels,   fmt='%i')
    np.savetxt(path + "data_test.out",      test_data,      fmt='%f')
    np.savetxt(path + "labels_test.out",    test_labels,    fmt='%i')
    print "'.out' files save to " + path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='generate a random classification problem')
    parser.add_argument('-c', '--complexity', metavar = 'complexity', type = int,
            default = 2, help = 'number of independent (informative) features')
    complexity = parser.parse_args().complexity

    mk_madelon(complexity, N)
