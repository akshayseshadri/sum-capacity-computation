import numpy as np
import scipy as sp
import bisect
import warnings

from scipy import stats

import project_root # noqa
from lipschitzlike_optimization.fill_simplex import get_grid_point

"""
    Maximization of Lipschitz-like functions over the standard simplex using dense curves.

    Author: Akshay Seshadri
"""

def maximize_lipschitzlike_function_standard_simplex_grid_search(f, beta, d, eps = 1e-1, max_iter = None, quiet = False):
    """
        Computes the maximum of the Lipschitz-like function f that satisfies
            
            |f(x) - f(y)| <= beta(||x - y||)

        when x, y belong to the standard simplex in d dimensions.

        The maximum is obtained by searching the grid

        \Delta_{d, N} = {n/N | n \in \mathbb{N}^d, \sum_{i = 1}^d n_i = N}

        where d is the dimension of the simplex and the integer N determines the spacing of the grid.

        Args:
            - f: Lipschitz-like function from D to R, where D \subseteq R^n is the domain of 'f'
            - beta: function defining the Lipschitz-like property of 'f'
            - d: ambient dimension of the domain of f
            - eps: precision to which the maximum is computed
            - max_iter: maximum number of global iterations to be allowed, meant as a safety check
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
        raise ValueError("Unable to find a 'delta' satisfying beta(delta) = eps/2.")

    # find the sum N of coordinates used to construct the integer grid
    N = int(np.ceil(1 / delta**2))

    # maximum number of iterations needed by the algorithm to converge
    N_maxiter = int(sp.special.comb(N + d - 1, d - 1, 'exact'))

    if max_iter == None:
        max_iter = N_maxiter
    elif max_iter < N_maxiter:
        warnings.warn("'max_iter' iterations not sufficient to search the grid. The output may not give the maximum to the desired precision.")

    max_iter = min([int(max_iter), N_maxiter])

    if not quiet:
        print("---------- Maximize Lipschitz-like function over standard simplex using grid search  ---------")
        print("Number of iterations needed to compute the maximum to the specified precision:", N_maxiter)

    # maximum value of f
    f_max = -np.inf

    # search the grid upto max_iter iterations
    for i in range(max_iter):
        # obtain the grid
        x_i = get_grid_point(i, N, d)

        # update the maximum value of f
        f_max = max([f(x_i), f_max])

    return f_max
