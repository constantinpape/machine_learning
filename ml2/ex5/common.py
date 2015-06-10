import numpy
import scipy.sparse
import scipy.sparse.linalg

def compute_alpha(train_x, train_y, kernel, kernel_params, tau):
    """Compute the alpha vector of the ridge regressor.

    :param train_x: training x data
    :param train_y: training y data
    :param tau: parameter tau of the ridge regressor
    :param sigma: parameter sigma of the gaussian kernel
    :return: alpha vector
    """
    print "building input kernel matrix"
    K = kernel(train_x, kernel_params)
    print "sparsity: %.2f%%" % ( 100. - (float(100*K.nnz) / (K.shape[0]*K.shape[1])) )
    M = K + tau * scipy.sparse.identity(train_x.shape[0])
    y = scipy.sparse.csc_matrix(train_y[:,None])
    print "solving sparse system"
    alpha = scipy.sparse.linalg.cg(M, train_y, maxiter = 1000)
    print "done computing alpha"
    return alpha[0]

def compute_beta(train_x, kernel, kernel_params, tau):
    K = kernel(train_x, kernel_params)
    M = K + tau * scipy.sparse.identity(train_x.shape[0])
    return scipy.sparse.linalg.inv(M)
