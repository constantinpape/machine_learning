import vigra
import numpy
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial

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

def modified_exponential_kernel(data, params):
    """Compute the modified exponential kernel matrix.

    :param data: data matrix
    :param params: paramters of the kernel
    :param params[0]: parameter gamma of the kernel
    :param params[1]: parameter rho of the kernel
    :param params[2]: parameter max_distance of the kernel
    :return: modified exponential kernel matrix
    """
    assert len(data.shape) == 2
    assert len(params)     == 3
    gamma           = params[0]
    rho             = params[1]
    max_distance    = params[2]

    factor = - 1. /  rho**gamma
    limit = numpy.exp(factor*max_distance**gamma)
    # Find the pairwise distances and compute the modified exponential kernel.
    K = []
    for k in data:
        d = numpy.exp(factor*numpy.sum(numpy.abs((data - k))**gamma,axis=1))
        d[d < limit] = 0.0  # truncate the modified exponential
        d = scipy.sparse.csc_matrix(d[:,None])
        K.append(d)
    K = scipy.sparse.hstack(K)
    return K

class ModifiedKernelRidgeRegressor(object):
    """Kernel Ridge Regressor with modified exponential kernel.
    """

    def __init__(self, tau, rho, gamma):
        self.dim = None
        self.train_x = None
        self.alpha = None
        self.mean_y = None
        self.std_y = None
        self.tau = tau
        self.rho = rho
        self.gamma = gamma
        self.max_distance = 4.0*rho
        self.scale = -1./rho**gamma
        print self.tau, self.rho, self.gamma

    def train(self, train_x, train_y):
        """Train the kernel ridge regressor.

        :param train_x: training x data
        :param train_y: training y data
        """
        assert len(train_x.shape) == 2
        assert len(train_y.shape) == 1
        assert train_x.shape[0] == train_y.shape[0]

        self.dim = train_x.shape[1]
        self.train_x = train_x.astype(numpy.float32)
        self.tree = scipy.spatial.cKDTree(self.train_x)

        self.mean_y = train_y.mean()
        self.std_y = train_y.std()
        train_y_std = (train_y - self.mean_y) / self.std_y

        params = (self.gamma, self.rho, self.max_distance)
        self.alpha = compute_alpha(
                self.train_x,
                train_y_std,
                modified_exponential_kernel,
                params,
                self.tau
                )

    def predict_single(self, pred_x):
        """Predict the value of a single instance.

        :param pred_x: x data
        :return: predicted value of pred_x
        """
        assert len(pred_x.shape) == 1
        assert pred_x.shape[0] == self.dim
        indices = numpy.asarray(self.tree.query_ball_point(pred_x, self.max_distance))
        dist = numpy.sum( numpy.absolute(self.train_x[indices]-pred_x)**self.gamma, axis=1)
        kappa = numpy.exp(self.scale*dist)
        pred_y = numpy.dot(kappa, self.alpha[indices])
        return self.std_y * pred_y + self.mean_y

    def predict(self, pred_x):
        """Predict the values of pred_x.

        :param pred_x: x data
        :return: predicted values of pred_x
        """
        assert len(pred_x.shape) == 2
        assert pred_x.shape[1] == self.dim
        pred_x = pred_x.astype(numpy.float32)
        return numpy.array([self.predict_single(x) for x in pred_x])
