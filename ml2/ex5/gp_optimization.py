import sys
import argparse
import vigra
import numpy
import time
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial

from ModifiedKernelRidgeRegressor   import ModifiedKernelRidgeRegressor
from MaternKernelRidgeRegressor     import MaternKernelRidgeRegressor

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

def kernel_ridge_regression(im, tau, rho, gamma):
    # copy the image
    im_ret = im.copy()

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
    print "predicting..."
    pred_y = r.predict(pred_x)
    print "done predicting"

    # Write the predicted values back into the image and show the result.
    im_ret[unknown_ind] = pred_y
    stop = time.time()
    print "Train and predict took %.02f seconds." % (stop-start)
    return im_ret

def matern_regression(Q, P, E, sig_rho, sig_gamma, sig_tau, lambd):

    # known points in the hyperparameter space
    known_x = numpy.array(P)
    known_y = numpy.array(E)

    r = MaternKernelRidgeRegressor(sig_rho, sig_gamma, sig_tau, lambd)
    r.train(known_x, known_y)
    return r.predict(Q)


def calc_mse(image, interpolation):
    assert image.shape == interpolation.shape
    width = float(image.shape[0])
    heigth = float(image.shape[1])
    return 1./( width * heigth) * numpy.sum( (image - interpolation)**2 )

def get_sobol(N):
    import sobol
    limit_rho_up = 10.
    limit_tau_up = 1.
    limit_gamma_up = 4.
    limit_rho_dn = 2.
    limit_tau_dn = 0.1
    limit_gamma_dn = 1.
    parameterLowerLimits = numpy.array([limit_rho_dn, limit_gamma_dn, limit_tau_dn])
    parameterUpperLimits = numpy.array([limit_rho_up, limit_gamma_up, limit_tau_up])
    Q = numpy.zeros( (N, 3) )
    for i in range(N):
        rho, gamma, tau = sobol.i4_sobol(3,i)[0] * (parameterUpperLimits - parameterLowerLimits) +parameterLowerLimits
        Q[i,0] = rho
        Q[i,1] = gamma
        Q[i,2] = tau
    return Q

def bayesian_hp_optimization(Q):
    im = numpy.squeeze(vigra.readImage("cc_90.png"))
    im_orig = numpy.squeeze(vigra.readImage("charlie-chaplin.jpg"))
    P = []
    E = []
    # initialize
    f = open("cache/bayes_opt.txt","w")
    start = time.time()
    for i in range(20):
        interpolation = kernel_ridge_regression( im, Q[i,2], Q[i,0], Q[i,1] )
        P.append( [Q[i,0], Q[i,1], Q[i,2]] )
        E.append( calc_mse(im_orig, interpolation) )
        # save result
        res = str(Q[i,0]) + str(" ") + str(Q[i,1]) + str(" ") + str(Q[i,2]) + str(" ") + str(E[i]) + '\n'
        f.write(res)
        f.flush()


    # TODO should we remove known vals from Q ?
    # remove known values from Q
    # Q = numpy.delete(Q, numpy.arange(20), axis=0)
    # parameter for the matern regression
    sig_rho     = 4.
    sig_gamma   = 1.
    sig_tau     = 1.
    lambd       = .3
    for i in range(20):
        mse, var = matern_regression(Q, P, E, sig_rho, sig_gamma, sig_tau, lambd)
        utility = numpy.divide( mse, numpy.sqrt(var) )
        best_hp = numpy.argmin(utility)
        P.append( Q[best_hp])
        interpolation = kernel_ridge_regression( ima, Q[best_hp,2], Q[best_hp,0], Q[best_hp,1])
        E.append( calc_mse(im_orig, interpolation))
        res = str(Q[best_hp,0]) + str(" ") + str(Q[best_hp,1]) + str(" ") + str(Q[best_hp,2]) + str(" ") + str(E[-1]) + '\n'
        f.write(res)
        f.flush()
    # TODO should we remove known vals from Q ?
    # remove known values from Q
    # Q = numpy.delete(Q, new_hp, axis=0)
    stop = time.time()
    print "Bayesian parameter optimization took %.02f seconds." % (stop-start)
    best_hp = numpy.argmin(E)
    f.close()
    return P[best_hp], E[best_hp]

def random_hp_optimization(Q):
    f = open("cache/rand_opt.txt","w")
    image = numpy.squeeze(vigra.readImage("cc_90.png"))
    im_orig = numpy.squeeze(vigra.readImage("charlie-chaplin.jpg"))
    start = time.time()
    P = []
    E = []
    N = Q.shape[0]
    rand_indices = numpy.random.randint(0, N, size = 40)
    for i in rand_indices:
        interpolation = kernel_ridge_regression( image, Q[i,2], Q[i,0], Q[i,1] )
        P.append( [Q[i,0], Q[i,1], Q[i,2]] )
        E.append( calc_mse(im_orig, interpolation) )
        res = str(Q[i,0]) + str(" ") + str(Q[i,1]) + str(" ") + str(Q[i,2]) + str(" ") + str(E[-1]) + '\n'
        f.write(res)
        f.flush()
    rand_hp = numpy.argmin(E)
    stop = time.time()
    print "Random parameter optimization took %.02f seconds." % (stop-start)
    f.close()
    return P[rand_hp], E[rand_hp]

def main():
    N_samples = 2000
    Q = get_sobol(N_samples)

    print "starting optimization with gaussian process"
    best_hp, best_mse = bayesian_hp_optimization(Q)
    print "Best hyperparameter found via Bayesian Optimization of HP:"
    print "Rho:", best_hp[0], "Gamma:", best_hp[1], "Tau:", best_hp[2]
    print "Resulting MSE:", best_mse

    #print "starting optimization with grid search"
    #rand_hp, rand_mse = random_hp_optimization(Q)
    #print "Best hyperparameter found via Random Optimization of HP:"
    #print "Rho:", rand_hp[0], "Gamma:", rand_hp[1], "Tau:", rand_hp[2]
    #print "Resulting MSE:", rand_mse

    return 0

def use_ridge_regression():
    # Read the command line arguments.
    args = process_command_line()
    im_orig = numpy.squeeze(vigra.readImage("cc_90.png"))
    im = kernel_ridge_regression(im_orig, args.tau, args.rho, args.gamma)
    vigra.impex.writeImage(im, "res.png")
    im_true = numpy.squeeze(vigra.readImage("charlie-chaplin.jpg"))
    print "MSE: ", calc_mse(im, im_true)

    return 0

def test_matern_reg():
    Q = get_sobol(100)
    P = [Q[0],Q[1],Q[2],Q[3],Q[4],Q[5]]
    E = [   1,   2,   3,   4,   5,  6 ]
    for i in range(10):
        mse, var = matern_regression(Q, P, E, 1, 1, 1, 0.3)
    print "Matern regression ran completely"
    return 0

if __name__ == "__main__":
    status = main()
    #status = test_matern_reg()
    #status = use_ridge_regression()
    sys.exit(status)
