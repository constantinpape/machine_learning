import vigra
import numpy
import scipy.sparse
import scipy.sparse.linalg

from common import compute_alpha, compute_beta

def matern_kernel(data, params):
    assert len(data.shape) == 2
    assert len(params)     == 4

    sig_rho     = params[0]
    sig_gamma   = params[1]
    sig_tau     = params[2]
    min_val     = params[3]

    # Find the pairwise distances and compute the matern kernel.
    K = []
    for k in data:
        s = ( ( data[:,0] - k[0] ) / sig_rho )**2
        + ( ( data[:,1] - k[1] ) / sig_gamma )**2
        + ( ( data[:,2] - k[2] ) / sig_tau )**2
        d = (1. + numpy.sqrt(5*s) + 5./3. * s)*numpy.exp(-numpy.sqrt(5*s))
        d[d<min_val] = 0.
        if not scipy.sparse.issparse(d):
            d = scipy.sparse.coo_matrix(d)
        K.append( d )
    K = scipy.sparse.vstack(K)
    return K

class MaternKernelRidgeRegressor(object):
    """Kernel Ridge Regressor with matern kernel.
    """

    def __init__(self, sig_rho, sig_gamma, sig_tau, lambd):
        self.dim = None
        self.train_x = None
        self.alpha = None
        self.mean_y = None
        self.std_y = None
        self.sig_rho = sig_rho
        self.sig_gamma = sig_gamma
        self.sig_tau = sig_tau
        self.lambd = lambd
        self.min_val = 1.e-5

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

        self.mean_y = train_y.mean()
        self.std_y = train_y.std()
        train_y_std = (train_y - self.mean_y) / self.std_y

        params = (self.sig_rho, self.sig_gamma, self.sig_tau, self.min_val)
        self.alpha = compute_alpha(self.train_x, train_y_std, matern_kernel, params, self.lambd)
        self.beta  = compute_beta(self.train_x, matern_kernel, params, self.lambd)

    def predict_single(self, pred_x):
        """Predict the value of a single instance.

        :param pred_x: x data
        :return: predicted value of pred_x
        """
        assert len(pred_x.shape) == 1
        assert pred_x.shape[0] == self.dim
        dist =   ( (self.train_x[:,0] - pred_x[0]) / self.sig_rho   )**2.
        dist +=  ( (self.train_x[:,1] - pred_x[1]) / self.sig_gamma )**2.
        dist +=  ( (self.train_x[:,2] - pred_x[2]) / self.sig_tau   )**2.
        kappa = (1 + numpy.sqrt(5*dist) + 5./3.*dist) * numpy.exp( -numpy.sqrt(5*dist) )
        kappa[kappa<self.min_val] = 0.
        pred_y = numpy.dot(kappa, self.alpha)
        var = 1. - kappa.transpose().dot(self.beta.dot(kappa))
        return self.std_y * pred_y + self.mean_y, var

    def predict(self, pred_x):
        """Predict the values of pred_x.

        :param pred_x: x data
        :return: predicted values of pred_x
        """
        assert len(pred_x.shape) == 2
        assert pred_x.shape[1] == self.dim
        pred_x = pred_x.astype(numpy.float32)
        means = []
        vars  = []
        for x in pred_x:
            mean, var = self.predict_single(x)
            means.append(mean)
            vars.append(var)
        return numpy.array( means ), numpy.array( vars )
