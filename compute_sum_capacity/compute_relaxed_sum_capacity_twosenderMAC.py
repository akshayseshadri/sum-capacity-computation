import numpy as np
import scipy as sp
import cvxpy as cp

import itertools

from scipy import special

"""
    Computes the capacity corresponding to the convex relaxation of sum capacity by dropping the product distribution constraint.

    Author: Akshay Seshadri
"""

def compute_relaxed_sum_capacity_twosenderMAC(N, d1, d2, quiet = False):
    """
        Finds an upper bound on the sum capacity of the MAC N(z | x1, x2).

        The exact sum capacity is given by

            S(N) = \max_{p1, p2} H(\sum_{x1, x2} N(z | x1, x2) p1(x1) p2(x2)) + \sum_{x1, x2} p1(x1) p2(x2) \sum_z N(z | x1, x2) log(N(z | x1, x2))

        where p1 is an element of (d1 - 1)-dimensional standard simplex while p2 is an element of (d2 - 1)-dimensional standard simplex.

        An upper bound on the sum capacity is obtained through a convex relaxation of the above (non-convex) optimization problem by dropping the product constraint.
        (the corresponding region is called relaxed Ahlswede-Liao region)

            S(N) <= \max_{p12} H(\sum_{x1, x2} N(z | x1, x2) p12(x1, x2)) + \sum_{x1, x2} p12(x1, x2) \sum_z N(z | x1, x2) log(N(z | x1, x2))

        The matrix for the channel N must be constructed such that N[z, (x1, x2)] corresponds to (x1, x2) obtained as per itertools.product for each fixed (row) z.
        That is, the colums of N must be arranged as per (0, 0), ..., (0, d2 - 1), ..., (d1 - 1, 0), ..., (d1 - 1, d2 - 1).

        Args:
            - N        : probability transition matrix for the MAC
                         N must be constructed such that N[z, (a1, a2)] corresponds to (a1, a2) obtained as per itertools.product for each fixed (row) z.
                         That is, the columns of N must be arranged as per (0, 0), ..., (0, d2 - 1), ..., (d1 - 1, 0), ..., (d1 - 1, d2 - 1).
            - d1       : size of the first alphabet
            - d2       : size of the second alphabet
            - quiet    : suppress printing of output
    """
    # the channel matrix (of shape |Z| x |X1||X2|)
    N = np.asarray(N)

    # size of Z
    d = N.shape[0]

    # ensure that d1 and d2 are integers
    d1, d2 = int(d1), int(d2)

    # ensure that d1 and d2 are consistent with the channel shape
    if not d1*d2 == N.shape[1]:
        raise ValueError("Channel shape and supplied dimensions d1 and d2 don't agree. N should be of shape d x d1d2.")

    # vector used in optimization
    B = np.array([np.sum(sp.special.entr(N[:, j1*d2 + j2])) for (j1, j2) in itertools.product(range(d1), range(d2))])

    # solve the outer optimization over p2
    # optimization variable
    p12 = cp.Variable(shape = (d1*d2, 1), nonneg = True)

    # constraint set for p12 is the standard (d1*d2 - 1)-dimensional simplex
    # since p12 has already been defined to be non-negative, we only require it to sum to 1
    constr_p12 = [cp.sum(p12) == 1]

    # objective function
    f = cp.Maximize(cp.sum(cp.entr(N @ p12)) - B @ p12)

    # optimization problem
    opt_prob = cp.Problem(f, constr_p12)

    # maximize the objective function
    opt_prob.solve()

    # relaxed sum capacity, which is an upper bound on the sum capacity
    C_N = opt_prob.value

    # p12 achieving the maximum f value
    p12_max = p12.value

    if not quiet:
        print("Upper bound on the sum capacity of the MAC obtained from convex relaxation:", C_N)

    return C_N
