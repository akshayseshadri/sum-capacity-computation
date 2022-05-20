import numpy as np
import scipy as sp
import bisect

from scipy import stats, optimize

import project_root # noqa
from lipschitzlike_optimization.fill_simplex import obtain_grid_curve_point
from lipschitzlike_optimization.maximize_lipschitzlike_function_interval import Maximize_Lipschitzlike_Function_Interval

"""
    Maximization of Lipschitz-like functions over the standard simplex using dense curves.

    Author: Akshay Seshadri
"""

def maximize_lipschitzlike_function_standard_simplex_dense_curve(f, beta, d, eps = 1e-1, max_iter = 1e3, quiet = False):
    """
        Computes the maximum of the Lipschitz-like function f that satisfies
            
            |f(x) - f(y)| <= beta(||x - y||)

        when x, y belong to the standard simplex in d dimensions.

        A dense curve to fill the standard simplex, so that the optimization is effectively one-dimensional.
        This resulting optimization is solved using modified Piyavskii-Schubert algorithm for Lipschitz-like optimization.

        Args:
            - f: Lipschitz-like function from D to R, where D \subseteq R^n is the domain of 'f'
            - beta: function defining the Lipschitz-like property of 'f'
            - d: ambient dimension of the domain of f
            - eps: precision to which the maximum is computed
            - max_iter: maximum number of global iterations to be allowed
            - quiet: suppress printing of output
    """
    # dimension of the problem
    d = int(d)

    # zero-dimensional simplex is simply the point 1
    if d == 1:
         return f(1)

    # compute delta from the given precision and dimension, such that beta(delta) <= epsilon/4
    # since beta_ext is a monotonically increasing function, we can compute alpha using root finding
    # make the precision slightly smaller than 0.25*eps to absorb any optimization errors
    delta, delta_root_res = sp.optimize.brentq(lambda x: beta(x) - (1/2 - 1e-3) * eps, 0, 10, full_output = True)

    # ensure that root finding algorithm has converged
    if delta_root_res.converged is False:
        raise ValueError("Unable to find a suitable 'delta' for given 'beta' and 'eps'.")

    # find the sum N of coordinates used to construct the integer grid
    N = int(np.ceil(2 * (d - 1) / delta))

    # maximum number of iterations neede by the algorithm to converge
    N_maxiter = np.ceil(2*sp.special.comb(N + d - 1, d - 1) / (N * delta))

    if not quiet:
        print("---------- Maximize Lipschitz-like function over standard simplex using dense curves ---------")
        print("Number of iterations for convergence in the worst case:", N_maxiter)

    # obtain the length of the curve
    q_max = sp.special.comb(N + d - 1, d - 1, 'exact') * 2/N

    # the curve satisfies the contractive property ||h(q) - h(q')||_1 <= \min(|q - q'|, 2)
    # therefore, we have beta_h(x) = \min(x, 2)
    # since every other p-norm for p >= 1 is smaller than 1-norm, we have ||h(q) - h(q')||_p <= \min(|q - q'|, 2)
    # so the specified norm doesn't really matter

    # construct the curve-parametrized function: g = f o h
    g = lambda q: f(obtain_grid_curve_point(q, N, d))

    # beta for g = f o h is beta o beta_h(x) = beta(\min(x, 2))
    beta_g = lambda x: beta(min([x, 2]))

    # compute the maximum of the h-parametrized extended function using 1D Lipschitz-like adaptation of Piyavskii-Schubert algorithm
    g_maximizer = Maximize_Lipschitzlike_Function_Interval(g, beta_g, 0, q_max, (1/2)*eps, max_iter, quiet)
    g_max = g_maximizer.maximize()

    return g_max
