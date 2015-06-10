import vigra
import numpy
import scipy.linalg

def matern_kernel(data, params):
    assert len(data.shape) == 2
    assert len(params)     == 3

    sig_rho     = params[0]
    sig_gamma   = params[1]
    sig_tau     = params[2]

    # Find the pairwise distances and compute the matern kernel.
    K = []
    for k in data:
        s = ( ( data[:,0] - k[0] ) / sig_rho )**2
        + ( ( data[:,1] - k[1] ) / sig_gamma )**2
        + ( ( data[:,2] - k[2] ) / sig_tau )**2
        d = (1. + numpy.sqrt(5*s) + 5./3. * s)*numpy.exp(-numpy.sqrt(5*s))
        K.append( d )
    K = numpy.vstack(K)
    print K.shape
    return K

def compute_alpha(train_x, kernel, kernel_params, tau):
    """Compute the alpha vector of the ridge regressor.

    :param train_x: training x data
    :param train_y: training y data
    :param tau: parameter tau of the ridge regressor
    :param sigma: parameter sigma of the gaussian kernel
    :return: alpha vector
    """
    print "building input kernel matrix"
    K = kernel(train_x, kernel_params)
    M = K + tau * numpy.eye(train_x.shape[0])
    invKern = scipy.linalg.inv(M)

    print "done computing alpha"
    return invKern

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
        self.train_y_std = (train_y - self.mean_y) / self.std_y

        params = (self.sig_rho, self.sig_gamma, self.sig_tau)
        self.alpha = compute_alpha(self.train_x, matern_kernel, params, self.lambd)

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
        pred_y = numpy.dot(kappa, self.alpha).dot(self.train_y_std)
        var = 1. - kappa.transpose().dot(self.alpha.dot(kappa))
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
