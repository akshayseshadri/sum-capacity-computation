import numpy as np
import scipy as sp
import cvxpy as cp

import itertools

from scipy import stats, optimize

import project_root # noqa
from lipschitzlike_optimization.maximize_lipschitzlike_function_interval import Maximize_Lipschitzlike_Function_Interval
from lipschitzlike_optimization.maximize_lipschitzlike_function_standard_simplex_dense_curve import maximize_lipschitzlike_function_standard_simplex_dense_curve
from lipschitzlike_optimization.maximize_lipschitzlike_function_standard_simplex_grid_search import maximize_lipschitzlike_function_standard_simplex_grid_search

"""
    Computes the sum capacity of any two sender MAC to the specified precision.

    Author: Akshay Seshadri
"""

def maximize_lipschitzlike_function_standard_simplex(f, beta, d, eps = 5e-2, max_iter = 1e3, alg = 'dense_curve', quiet = False):
    """
        Maximize the beta-Lipschitz-like function f over the standard simplex in d dimensions as per the specified algorithm.

        The norm used throughout is l1 norm.

        When d = 2 and algorithm is 'dense_curve', standard simplex is just an interval so we optimize over this interval.

        Args:
            - f        : objective function to be maximized
            - beta     : function defining Lipschitz-like property of f
            - d        : dimension of the problem
            - eps      : tolerance
            - max_iter : maximum number of outer iterations allowed
            - alg      : algorithm used to perform the Lipschitz-like optimization
                         valid options are "dense_curve" (default) and "grid_search"
            - quiet    : suppress printing of output
    """
    # ensure that alg is specified in lower case
    alg = str(alg).lower()
    if not alg in ['dense_curve', 'grid_search']:
        raise ValueError("Please specify a valid algorithm.")

    # ensure that the dimension d is an integer
    d = int(d)

    # if d = 2 and we are using dense_curve algorithm, then optimize over the interval [0, 1]
    # the norm ||vec(x) - vec(y)||_1 = 2 |x - y|, writing vec(x) = (x, 1 - x) and vec(y) = (y, 1 - y) for vec(x), vec(y) in standard simplex of dimension 2
    # therefore, beta(||vec(x) - vec(y)||_1) needs to be changed to beta(2|x - y|)
    if d == 2 and alg == 'dense_curve':
        f_maximizer = Maximize_Lipschitzlike_Function_Interval(f, beta, 0, 1, (1/2)*eps, max_iter)
        f_max = f_maximizer.maximize()
    else:
        if alg == 'dense_curve':
            f_max = maximize_lipschitzlike_function_standard_simplex_dense_curve(f, beta, d, eps, max_iter, quiet)
        elif alg == 'grid_search':
            f_max = maximize_lipschitzlike_function_standard_simplex_grid_search(f, beta, d, eps, max_iter, quiet)

    return f_max

def compute_sum_capacity_twosenderMAC(N, d1, d2, eps = 5e-2, max_iter = 1e3, alg = 'dense_curve', quiet = False):
    """
        Given a two-sender one-receiver MAC N, with input alphabet sizes d1 and d2, finds the sum capacity of N.

        The sum capacity is given as

            S(N) = \max_{p, q} H(\sum_{a1, a2} N(z | a1, a2) p(a1) q(a2)) + \sum_{a1, a2} p(a1) q(a2) \sum_z N(z | a1, a2) log(N(z | a1, a2))

        where p1 is an element of (d1 - 1)-dimensional standard simplex while p2 is an element of (d2 - 1)-dimensional standard simplex. 

        Args:
            - N        : probability transition matrix for the MAC
                         N must be constructed such that N[z, (a1, a2)] corresponds to (a1, a2) obtained as per itertools.product for each fixed (row) z.
                         That is, the columns of N must be arranged as per (0, 0), ..., (0, d2 - 1), ..., (d1 - 1, 0), ..., (d1 - 1, d2 - 1).
            - d1       : size of the first alphabet
            - d2       : size of the second alphabet
            - eps      : tolerance
            - max_iter : maximum number of outer iterations allowed
            - alg      : algorithm used to perform the Lipschitz-like optimization
                         valid options are "dense_curve" (default) and "grid_search"
            - quiet    : suppress printing of output
    """
    # ensure that alg is specified in lower case
    alg = str(alg).lower()
    if not alg in ['dense_curve', 'grid_search']:
        raise ValueError("Please specify a valid algorithm.")

    # the channel matrix (of shape |Z| x |A1||A2|)
    N = np.asarray(N)

    # size of Z
    d = N.shape[0]

    # ensure that d1 and d2 are integers
    d1, d2 = int(d1), int(d2)

    # ensure that d1 and d2 are consistent with the channel shape
    if not d1*d2 == N.shape[1]:
        raise ValueError("Channel shape and supplied dimensions d1 and d2 don't agree. N should be of shape d x d1d2.")

    # rearrange the matrix N so that d2 <= d1 always
    N_rearraged = np.zeros_like(N)
    if d1 < d2:
        for (j1, j2) in itertools.product(range(d1), range(d2)):
            N_rearraged[:, j2*d1 + j1] = N[:, j1*d2 + j2]
        # swap d1 and d2
        d1, d2 = d2, d1
        # update N
        N = N_rearraged.copy()

    # entropy of channel at each input pair
    B = np.array([np.sum(sp.special.entr(N[:, j1*d2 + j2])) for (j1, j2) in itertools.product(range(d1), range(d2))])

    # maximum value of entropy of channel over input pairs
    H_N_max = np.max(B)

    ### inner optimization over p for a fixed q
    # optimization variable
    p = cp.Variable(shape = (d1, 1), nonneg = True)

    # parameters for the optimization
    A_q = cp.Parameter(shape = (d, d1))
    b_q = cp.Parameter(shape = (1, d1))

    # constraint set for p is the standard (d1 - 1)-dimensional simplex
    # since p has already been defined to be non-negative, we only require it to sum to 1
    constr_p = [cp.sum(p) == 1]

    # objective function is the mutual information
    I_pq = cp.Maximize(cp.sum(cp.entr(A_q @ p)) - b_q @ p)

    # optimization problem
    opt_prob_p = cp.Problem(I_pq, constr_p)

    # initialization
    q = np.ones(d2)/d2
    A_q.value = np.hstack([N[:, j1*d2 + np.arange(d2)].dot(q).reshape((d, 1)) for j1 in range(d1)])
    b_q.value = np.array([B[j1*d2 + np.arange(d2)].dot(q) for j1 in range(d1)]).reshape((1, d1))

    # function x * log(y) for computing binary entropy
    xlogy = sp.special.xlogy

    if d2 == 2 and alg == 'dense_curve':
        # function that computes the optimum over the (d1 - 1)-dimensional simplex
        def f(s):
            """
                Computes

                    \max_p H(A_q p) - <b_q, p>

                where A_q(z, a1) = \sum_{a2} N(z | a1, a2) q(a2)
                and   b_q(a1) = \sum_{a2} q(a2) \sum_z N(z | a1, a2) log(N(z | a1, a2))
            """
            q = [s, 1 - s]
            A_q.value = np.hstack([N[:, j1*d2 + np.arange(d2)].dot(q).reshape((d, 1)) for j1 in range(d1)])
            b_q.value = np.array([B[j1*d2 + np.arange(d2)].dot(q) for j1 in range(d1)]).reshape((1, d1))

            opt_prob_p.solve(warm_start = True)

            return opt_prob_p.value

        # function defining the Lipschitz-like property of f
        def beta(x):
            # heaviside step function is used instead of if-else statement for defining \overline{h}
            xleqhalf = np.heaviside(0.5 - x, 1)
            return 2 * (0.5*np.log(d - 1) + H_N_max) * x - xleqhalf * (xlogy(x, x) + xlogy(1 - x, 1 - x)) + (1 - xleqhalf) * np.log(2)
    else:
        # function that computes the optimum over the (d1 - 1)-dimensional simplex
        def f(q):
            """
                Computes

                    \max_p H(A_q p) - <b_q, p>

                where A_q(z, a1) = \sum_{a2} N(z | a1, a2) q(a2)
                and   b_q(a1) = \sum_{a2} q(a2) \sum_z N(z | a1, a2) log(N(z | a1, a2))
            """
            A_q.value = np.hstack([N[:, j1*d2 + np.arange(d2)].dot(q).reshape((d, 1)) for j1 in range(d1)])
            b_q.value = np.array([B[j1*d2 + np.arange(d2)].dot(q) for j1 in range(d1)]).reshape((1, d1))

            opt_prob_p.solve(warm_start = True)

            return opt_prob_p.value

        # function defining the Lipschitz-like property of f
        def beta(x):
            xb2 = 0.5 * x
            # heaviside step function is used instead of if-else statement for defining \overline{h}
            xleq1 = np.heaviside(0.5 - xb2, 1)
            return (0.5*np.log(d - 1) + H_N_max) * x - xleq1 * (xlogy(xb2, xb2) + xlogy(1 - xb2, 1 - xb2)) + (1 - xleq1) * np.log(2)

    # compute the sum capacity by performing the optimization
    S_N = maximize_lipschitzlike_function_standard_simplex(f, beta, d2, eps, max_iter, alg, quiet)

    return S_N
