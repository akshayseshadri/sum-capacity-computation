import numpy as np
import scipy as sp
import cvxpy as cp

import itertools

import project_root # noqa
from compute_sum_capacity.compute_sum_capacity_twosenderMAC import compute_sum_capacity_twosenderMAC
from compute_sum_capacity.compute_relaxed_sum_capacity_twosenderMAC import compute_relaxed_sum_capacity_twosenderMAC

def compute_sum_capacity_NF_examples(example = 1, eps = 2e-2, max_iter = 1e3, alg = 'dense_curve', quiet = False):
    """
        Computes the sum capacity for two examples of the noise-free subspace MAC N_F given below.

        Example 0: N_F = [[1, 0.5, 0.5, 0.5],
                          [0, 0.5, 0.5, 0.5]]
                    
                   S(N_F) = h(4/5) - (2/5) ln(2) approx 0.223 nats
                   C(N_F) = h(4/5) - (2/5) ln(2) approx 0.223 nats

        Example 1: N_F = [[1, 0.5, 0.5, 0],
                          [0, 0.5, 0.5, 1]]
                    
                   S(N_F) = 0.5 ln(2) approx 0.3466 nats
                   C(N_F) = ln(2) approx 0.693 nats
    """
    # ensure that example is an integer
    example = int(example)

    # construct the channel matrix for the specified example
    if example == 0:
        N = np.array([[1, 0.5, 0.5, 0.5],\
                      [0, 0.5, 0.5, 0.5]])
    elif example == 1:
        N = np.array([[1, 0.5, 0.5, 0],\
                      [0, 0.5, 0.5, 1]])
    else:
        raise ValueError("Please provide a valid entry for 'example'")

    # compute the sum capacity of the specified MAC
    S_N = compute_sum_capacity_twosenderMAC(N, 2, 2, eps, max_iter, alg, quiet)

    # compute the relaxed sum capacity of the specified MAC
    C_N = compute_relaxed_sum_capacity_twosenderMAC(N, 2, 2, quiet)

    return (S_N, C_N)

def compute_sum_capacity_nBS_MAC(delta = 0, epsilon = 0, eps = 2e-2, max_iter = 1e3, alg = 'dense_curve', quiet = False):
    """
        Computes the sum capacity of noisy binary-switching (nBS) MAC.

        The channel matrix of nBS-MAC is given as
        N = [[delta/2,   delta/2,   1 - epsilon, epsilon    ]
             [delta/2,   delta/2,   epsilon,     1 - epsilon]
             [1 - delta, 1 - delta, 0,           0          ]]

        It has two binary inputs (d1 = d2 = 2) and one ternary output (d = 3).

        For delta = epsilon = 0, we get the binary-switching MAC (i.e., noise-free case).

        The convex relaxation is known to be tight, i.e., relaxed sum capacity matches the exact sum capacity (see Calvo & Fonollosa, 2010).
    """
    # construct the channel matrix
    N = np.array([[delta/2,   delta/2,   1 - epsilon, epsilon    ],
                  [delta/2,   delta/2,   epsilon,     1 - epsilon],
                  [1 - delta, 1 - delta, 0,           0          ]])

    # compute the sum capacity of nBS-MAC
    S_N = compute_sum_capacity_twosenderMAC(N, 2, 2, eps, max_iter, alg, quiet)

    # compute the relaxed sum capacity of nBS-MAC
    C_N = compute_relaxed_sum_capacity_twosenderMAC(N, 2, 2, quiet)

    return (S_N, C_N)

def compute_sum_capacity_random_MAC(d1 = 10, d2 = 2, do = 20, eps = 0.3, max_iter = 1e3, alg = 'dense_curve', quiet = False, seed = 10):
    """
        Computes the sum capacity of a randomly generated 2-sender MAC with input alphabet sizes d1, d2 and output alphabet size do.
    """
    # ensure that the dimensions d1, d2, do are integers
    d1, d2, do = int(d1), int(d2), int(do)

    # initialize a random number generator with the given seed
    rng = np.random.default_rng(int(seed))

    # randomly obtain d1*d2 probability vectors of size do
    p_list = [rng.random(do) for _ in range(d1*d2)]
    p_list = [p/sum(p) for p in p_list]

    # stack the probability vectors appropriately to get the probability transition matrix for the MAC
    N = np.vstack(p_list).T

    # compute the sum capacity of specified MAC
    S_N = compute_sum_capacity_twosenderMAC(N, d1, d2, eps, max_iter, alg, quiet)

    # compute the relaxed sum capacity of the specified MAC
    C_N = compute_relaxed_sum_capacity_twosenderMAC(N, d1, d2, quiet)

    return (S_N, C_N)

def get_MAC_nonlocal_game(W, d1, d2, a1, a2):
    """
        Given the winning condition W for a two player promise-free nonlocal game G with d1 questions for Alice and d2 questions for Bob, construct the MAC N_G described below.

        The output set is Z = X1 x X2, so that |Z| = d = d1 d2.

        The size of the input for Alice is |X1 x Y1| = d1 a1, while the size of the input for Bob is |X2 x Y2| = d2 a2.

        N_G(z | x1, x2) = \delta_{x1', x1} \delta{x2', x2} if (x1, x2) \in W
                        = 1/d                              otherwise

        The columns of N_G are arranged as per (x1, x2) obtained according to itertools.product.
    """
    # size of input alphabets
    d1, d2 = int(d1), int(d2)

    # size of output alphabet
    d = d1*d2

    # MAC N_G obtained from the promise-free nonlocal game G with winning condition W
    N_G = np.zeros((d, d1*a1*d2*a2))
    for (x1y1, x2y2) in itertools.product(range(d1*a1), range(d2*a2)):
        # range(di*ai) is arranged in the order xi*ai + yi for (xi, yi) in itertools.product(range(di), range(ai))
        # so we extract (xi, yi) from an element of range(di*ai)
        y1 = int(x1y1 % a1)
        x1 = int((x1y1 - y1) / a1)

        y2 = int(x2y2 % a2)
        x2 = int((x2y2 - y2) / a2)

        if ((x1, y1), (x2, y2)) in W:
            # generate the probability distribution with 1 at (x1, x2) and 0 elsewhere
            p_x1x2 = np.zeros(d)
            p_x1x2[x1*d2 + x2] = 1

            # the channel outputs the question pair with probability 1
            N_G[:, x1*d2 + x2] = p_x1x2
        else:
            # channel uniformly chooses a question pair and outputs that
            N_G[:, x1*d2 + x2] = np.ones(d) / d

    return N_G

def compute_sum_capacity_MAC_signalling_game(d1, d2, eps = 2e-2, max_iter = 1e3, alg = 'dense_curve', quiet = False):
    """
        Computes the sum capacity of the MAC obtained from the signalling game using classical strategies.

        In the signalling game, Alice and Bob receive questions from X_1 and X_2, respectively.
        They win the game if they infer the question received by the other person.
        That is, Y_1 = X_2 and Y_2 = X_1, and W = {((x_1, x_2), (y_1, y_2)) \in (X_1 x X_2) x (Y_1 x Y_2) | y_1 = x_2, y_2 = x_1}.

        args:
        - d1: size of X1
        - d2: size of X2
    """
    # number of questions given to Alice and Bob, respectively
    d1, d2 = int(d1), int(d2)

    if d1 <= 1 or d2 <= 1:
        raise ValueError("'d1' and 'd2' must be at least 2")

    # winning condition for the signalling game
    W = [(x, x[::-1]) for x in itertools.product(*[range(d1), range(d2)])]

    # MAC obtained from the signalling game
    N = get_MAC_nonlocal_game(W, d1, d2, d2, d1)

    # compute the sum capacity of the MAC obtained from the signalling game
    S_N = compute_sum_capacity_twosenderMAC(N, d1*d2, d2*d1, eps, max_iter, alg, quiet)

    # compute the relaxed sum capacity of the the MAC obtained from the signalling game
    C_N = compute_relaxed_sum_capacity_twosenderMAC(N, d1*d2, d2*d1, quiet)

    return (S_N, C_N)
