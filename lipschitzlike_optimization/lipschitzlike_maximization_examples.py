import numpy as np
import scipy as sp

from scipy import optimize

import project_root # noqa
from lipschitzlike_optimization.maximize_lipschitzlike_function_standard_simplex_dense_curve import maximize_lipschitzlike_function_standard_simplex_dense_curve
from lipschitzlike_optimization.maximize_lipschitzlike_function_standard_simplex_grid_search import maximize_lipschitzlike_function_standard_simplex_grid_search

"""
    Tests the Lipschitz-like maximization algorithm over the standard simplex using dense curves and grid search

    Author: Akshay Seshadri
"""

def maximize_sine_simplex(d = 3, eps = 0.1, max_iter = 1e3, quiet = True):
    """
        Maximize the sine of the Euclidean norm ||x||_2 for x in the standard simplex in d dimensions.

        Args:
            - d: dimension of the ambient Euclidean space
            - eps: precision to which the maximum is computed
            - max_iter: maximum number of global iterations to be allowed
            - quiet: suppress printing of output
    """
    # dimension of the system
    d = int(d)

    # objective function
    f = lambda x: np.sin(np.linalg.norm(x, ord = 2))

    # Lipschitzlike function for the objective
    beta = lambda x: x

    # hyper-rectangle bounding the standard simplex
    bounding_rectangle = [(0, 1)]*d

    # find the maximum of the objective using Lipschitzlike maximization over the standard simplex with dense curves
    f_max_standard_simplex_dense_curve = maximize_lipschitzlike_function_standard_simplex_dense_curve(f, beta, d, eps, max_iter, quiet)

    # find the maximum of the objective using Lipschitzlike maximization over the standard simplex by perfoming a grid search
    f_max_standard_simplex_grid_search = maximize_lipschitzlike_function_standard_simplex_grid_search(f, beta, d, eps, max_iter, quiet)

    print("True maximum:", np.sin(1))
    print("Maximum obtained using dense curve:", f_max_standard_simplex_dense_curve)
    print("Maximum obtained using grid search:", f_max_standard_simplex_grid_search)

def maximize_polynomial_simplex(p, d = 3, eps = 0.1, max_iter = 1e3, quiet = True):
    """
        Maximize the function f(x) = p(||x||_2) over the standard simplex in d dimensions,
        where p(x) = \sum_{i = 0}^n p_i x^i is the specified polynomial and ||x||_2 is the Euclidean norm of x.

        Args:
            - p: a vector of coefficients [p_n, ..., p_0] specifying a polynomial of degree n
            - d: dimension of the ambient Euclidean space
            - eps: precision to which the maximum is computed
            - max_iter: maximum number of global iterations to be allowed
            - quiet: suppress printing of output
    """
    # dimension of the system
    d = int(d)

    # objective function
    f = lambda x: np.polyval(p, np.linalg.norm(x, ord = 2))

    # Lipschitz constant of the polynomial p
    # the monomial x^i has a Lipschitz constant of i b^(i - 1) if 0 <= a <= x <= b
    # the 2-norm satisfies 1/\sqrt{d} <= ||x||_2 <= 1 for x in the standard simplex in d dimensions
    L_p = np.dot(np.arange(start = len(p) - 1, stop = -1, step = -1), np.abs(p))

    # Lipschitzlike function for the objective
    beta = lambda x: L_p*x

    # hyper-rectangle bounding the standard simplex
    bounding_rectangle = [(0, 1)]*d

    # true maximum of f is obtained by computing the maximum of p(x) over [1/sqrt(d), 1]
    z_max_true = sp.optimize.minimize_scalar(lambda x: -np.polyval(p, x), bounds = [1/np.sqrt(d), 1], method = 'bounded').x
    f_max_true = np.polyval(p, z_max_true)

    # find the maximum of the objective using Lipschitzlike maximization over the standard simplex with dense curves
    f_max_standard_simplex_dense_curve = maximize_lipschitzlike_function_standard_simplex_dense_curve(f, beta, d, eps, max_iter, quiet)

    # find the maximum of the objective using Lipschitzlike maximization over the standard simplex by perfoming a grid search
    f_max_standard_simplex_grid_search = maximize_lipschitzlike_function_standard_simplex_grid_search(f, beta, d, eps, max_iter, quiet)

    print("True maximum:", f_max_true)
    print("Maximum obtained using dense curve:", f_max_standard_simplex_dense_curve)
    print("Maximum obtained using grid search:", f_max_standard_simplex_grid_search)
