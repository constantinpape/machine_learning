import sys
import argparse
import vigra
import numpy
import time
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial

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
    gamma           = params[0]
    rho             = params[1]
    max_distance    = params[2]

    factor = - 1. /  rho**gamma
    limit = numpy.exp(factor*max_distance**gamma)
    # Find the pairwise distances and compute the modified exponential kernel.
    K = []
    for k in data:
        d = numpy.exp(factor*numpy.sum(np.abs((data - k))**gamma,axis=1))
        d[d < limit] = 0.0  # truncate the modified exponential
        d = scipy.sparse.csc_matrix(d[:,None])
        K.append(d)
    K = scipy.sparse.hstack(K)
    return K

def matern_kernel(data, sig_rho, sig_gamma, sig_tau):
    assert len(data.shape) == 2

    # Find the pairwise distances and compute the matern kernel.
    K = []
    for k in data:
        s = ( ( data[:,0] - k[:,0] ) / sig_rho )**2 + ( ( data[:,1] - k[:,1] ) / sig_gamma )**2 + ( ( data[:,2] - k[:,2] ) / sig_tau )**2
        d = (1. + np.sqrt(5*s) + 5./3. * s)*np.exp(-np.sqrt(5*s))
        K.append(d)
    K = hstack(K)
    return K

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
    print "sparsity: %.2f%%" % (float(100*K.nnz) / (K.shape[0]*K.shape[1]))
    M = K + tau * scipy.sparse.identity(train_x.shape[0])
    y = scipy.sparse.csc_matrix(train_y[:,None])
    print "solving sparse system"
    alpha = scipy.sparse.linalg.cg(M, train_y)
    print "done computing alpha"
    return alpha[0]

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
        self.alpha = compute_alpha(self.train_x, train_y_std, gaussian_kernel, params, self.tau)

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


def kernel_ridge_regression(tau, rho, gamma):
    # Load the image.
    im_orig = numpy.squeeze(vigra.readImage("cc_90.png"))

    # Make a copy, so both the original and the regressed image can be shown afterwards.
    im = numpy.array(im_orig)

    # Find the known pixels and the pixels that shall be predicted.
    known_ind = numpy.where(im != 0)
    unknown_ind = numpy.where(im >= 0)
    known_x = numpy.array(known_ind).transpose()
    known_y = numpy.array(im[known_ind])
    pred_x = numpy.array(unknown_ind).transpose()

    # Train and predict with the given regressor.
    start = time.time()
    print "training..."
    r = ModifiedKernelRidgeRegressor(tau, rho, gamma)
    r.train(known_x, known_y)
    print "done training"
    # pickle.dump(r, open("regressor.p", "wb"))
    # r = pickle.load(open("regressor.p", "rb"))
    print "predicting..."
    pred_y = r.predict(pred_x)
    print "done predicting"

    # Write the predicted values back into the image and show the result.
    im[unknown_ind] = pred_y
    stop = time.time()
    print "Train and predict took %.02f seconds." % (stop-start)
    vigra.impex.writeImage(im, "res.png")

def process_command_line():
    """Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Machine Learning exercise 5.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--tau", type=float, default=0.8,
                        help="parameter tau for ridge regression")
    parser.add_argument("-r", "--rho", type=float, default=7.5,
                        help="parameter rho of the modified_exponential_kernel")
    parser.add_argument("-g", "--gamma", type=float, default=1.7,
                        help="parameter gamma of the modified_exponential_kernel")

    return parser.parse_args()


def main():
    """Call the exercises.
    """
    # Read the command line arguments.
    args = process_command_line()
    kernel_ridge_regression(args.tau, args.rho, args.gamma)
    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
