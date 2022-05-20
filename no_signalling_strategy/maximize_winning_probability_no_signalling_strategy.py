import numpy as np
import scipy as sp
import cvxpy as cp
import itertools

"""
    Finds the maximum winning probability of the specified nonlocal game using no-signalling (NS) strategies.

    The same procedure can be used to maximize concave functions over NS strategies.

    Author: Akshay Seshadri
"""

def maximize_winning_prob_nonlocal_game(X_list, Y_list, W, pi, print_result = False):
    """
        Maximizes the winning probability of the nonlocal game G defined by X_list, Y_list, and W, when the questions are drawn as per pi.

        G is an N-player game with question sets X_1, ..., X_N and answer sets Y_1, ..., Y_N.
        The actual symbols of the questions and answers are inconsequential, so we just use |X_i| and |Y_i| for each i = 1, ..., N.
        The winning condition is described by the set W \subseteq (X_1 x ... x X_N) x (Y_1 x ... x Y_N).
        pi is the distribution over the questions, i.e., pi is an element of the probability simplex in R^{|X_1 x ... x X_N|}.

        args:
        - X_list : list of number of questions for each player, i.e, X_list = [|X_1|, ..., |X_N|]
        - Y_list : list of number of answers for each player, i.e, Y_list = [|Y_1|, ..., |Y_N|]
        - W      : winning condition to be specified as a list of tuples ((x_1, ..., x_N), (y_1, ..., y_N)),
                   where 1 <= x_i <= |X_i| and 1 <= y_i <= |Y_i| (i = 1, ..., N) point to the relevant question and answers that win the game.
                   Take note of the specific format for specifying the tuples.
        - pi     : element of the probability simplex in R^{|X_1 x ... x X_N|}, with elements order as per itertools.product(X_1, ..., X_N)

        - print_result: print the maximum winning probability and the optimal strategy in a human-readable format

        Remarks: Since pi is an arbitrary probability distribution over the questions, this can be used to enforce a promise set.

        The algorithm is based on the proof of Proposition 11 of the article
        "On the separation of correlation-assisted sum capacities of multiple access channels"
    """
    # for convenience, we will refer to |X_i| by Xi, |Y_i| by Yi, |X_1 x ... x X_N| by X, and |Y_1 x ... Y_N| by Y.
    # size of the question set
    X = int(np.prod(X_list))
    # size of the answer set
    Y = int(np.prod(Y_list))
    # number of players
    N = len(X_list)

    if not len(X_list) == len(Y_list):
        raise ValueError("Equal number of question and answer sets must be provided")

    # suppose that we fix the ordering of X_1 x ... x X_N given by itertools.product(X_1, ..., X_N)
    # then we can map (x_1, ..., x_n) \in X_1 x ... x X_N to the index 'i' of the corresponding element in itertools.product using
    # i = x_1 * (X_2*...*X_N) + x_2 * (X_3*...*X_N) + ... + x_{N - 1} * X_N + x_N
    # since we use cumulative products of X_i, we obtain [X_2*...*X_N, ..., X_{N - 1}*X_N, X_N, 1]
    X_cumprod_reversed = np.cumprod(X_list[:0:-1])[::-1].tolist() + [1]
    # mapping from (x_1, ..., x_N) to index in itertools.product:
    i_X_product = lambda x: sum([x[i]*X_cumprod_reversed[i] for i in range(N)])

    # similarly, we find a mapping to indices for Y = Y_1 x ... x Y_N
    Y_cumprod_reversed = np.cumprod(Y_list[:0:-1])[::-1].tolist() + [1]
    i_Y_product = lambda y: sum([y[i]*Y_cumprod_reversed[i] for i in range(N)])

    # range of X_j for each j
    X_range_list = [range(Xi) for Xi in X_list]

    # range of Y_j for each j
    Y_range_list = [range(Yi) for Yi in Y_list]

    # initialize the optimization problem
    v = cp.Variable((X*Y, 1), nonneg = True)

    ### constraints
    # conditional distribution constraint:
    # v = (v^(1), ..., v^(X)) where v^(i) is an element of the probability simplex in R^{|Y|}
    prod_simplex_constr = [cp.sum(v[i*Y: (i + 1)*Y]) == 1 for i in range(X)]

    # no-signalling constraints:
    # for each i = 1, ...., N, for each x_i \in X_i and y_i \in Y_i, we need
    # D_{x_i} S_{(x_i, y_i)} v = 0
    # define I_{x_i} = {(x_1', ..., x_i, ..., x_N') \in X | x_j' \in X_j, j \neq i}
    # and    I_{y_i} = {(y_1', ..., y_i, ..., y_N') \in Y | y_j' \in Y_j, j \neq i}
    # D_{x_i} is a (|I_{x_i}| - 1) x |I_{x_i}| matrix given as
    # [[1, -1, 0, ..., 0], [0, 1, -1, ..., 0], ..., [0, 0, 0, ..., -1]]
    # S_{(x_i, y_i)} is a |I_{x_i}| x |X||Y| matrix with kth row defined as the block matrix
    # [0_{1 x |Y|} ... s^T_{y_i} ... 0_{1 x |Y|}], with s^T_{y_i} a row vector of size |Y| in location k \in I_{x_i}
    # The vector s_{y_i} is of size |Y| with entries 1 at locations (y_1', ..., y_N') \in Y with y_i' = y_i and zero elsewhere
    constr_no_signalling = list()

    for i in range(N):
        # range for X_j for j \neq i
        X_range_list_jni = X_range_list[0: i] + X_range_list[i + 1: ]
        # range for Y_j for j \neq i
        Y_range_list_jni = Y_range_list[0: i] + Y_range_list[i + 1: ]

        # elements (tuples) specified by X_range_list_jni
        X_range_list_jni_elts = [list(x_jni) for x_jni in itertools.product(*X_range_list_jni)]
        # elements (tuples) specified by Y_range_list_jni
        Y_range_list_jni_elts = [list(y_jni) for y_jni in itertools.product(*Y_range_list_jni)]

        # D_{x_i} is of size (|I_xi| - 1) x |I_xi|, where |I_xi| = |X| / |Xi|
        # D_{x_i} actually only depends on i and not any particular element x_i \in X_i
        I_xi_size = int(X/X_list[i])
        D_xi = np.zeros((I_xi_size - 1, I_xi_size))
        for l in range(I_xi_size - 1):
            D_xi[l, l: l + 2] = [1, -1]

        for xi in range(X_list[i]):
            # define the index sets I_{x_i} and I_{y_i}
            # note that |I_{x_i}| = \prod_{j \neq i} |X_j| and |I_{y_i}| = \prod{j \neq i} |Y_j|
            # I_xi lists the indices of (x_1', ..., x_i, ..., x_N') with x_j' \in X_j for j \neq i
            I_xi = [i_X_product(x_jni[0: i] + [xi] + x_jni[i: N]) for x_jni in X_range_list_jni_elts]

            for yi in range(Y_list[i]):
                # I_yi lists the indices of (y_1', ..., y_i, ..., y_N') with y_j' \in Y_j for j \neq i
                I_yi = [i_Y_product(y_jni[0: i] + [yi] + y_jni[i: N]) for y_jni in Y_range_list_jni_elts]

                # s_yi is a |Y|-dimensional vector with 1 at indices specified by I_yi
                s_yi = np.zeros(Y)
                s_yi[I_yi] = 1

                # construct the matrix S_{(x_i, y_i)}
                # the kth row of S_{(x_i, y_i)} is [0_{1 x |Y|} ... s^T_{y_i} ... 0_{1 x |Y|}]
                S_xiyi = np.zeros((I_xi_size, X*Y))
                for k in range(I_xi_size):
                    S_xiyi[k] = np.concatenate([np.zeros(Y)]*I_xi[k] + [s_yi] + [np.zeros(Y)]*(X - (I_xi[k] + 1)))

                # the matrix A_{(x_i, y_i)} obtained as a product of D_{x_i} and S_{(x_i, y_i)}
                A_xiyi = D_xi.dot(S_xiyi)

                # the no-signalling constraint corresponding to the elements x_i \in X_i, y_i \in Y_i
                constr_no_signalling = constr_no_signalling + [A_xiyi @ v == np.zeros((I_xi_size - 1, 1))]

    constr = prod_simplex_constr + constr_no_signalling

    ### objective and maximization
    # objective: winning probability given as \sum_{(x, y) \in W} \pi(x) * p_{Y | X}(y | x) = \sum_{(x, y) \in W} pi(i(x)) * v^(i(x))[i(y)]
    f = cp.Maximize(cp.sum([pi[i_X_product(x)] * v[i_X_product(x)*Y + i_Y_product(y)] for (x, y) in W]))

    # optimization problem
    opt_prob = cp.Problem(f, constr)

    # maximize the winning probability
    opt_prob.solve()
    # maximum value of winning probability when the questions are drawn as per pi
    omegaG_pi = opt_prob.value
    # strategy achieving the maximum winning probability
    v_max = v.value

    if print_result:
        print("omega(G, pi) = %0.6f\n" %omegaG_pi)

        # output v_max in a human-readable format
        p_Y_X_max = ""
        for (i, x) in enumerate(itertools.product(*X_range_list)):
            v_max_i = v_max[i*Y: (i + 1)*Y]
            for (j, y) in enumerate(itertools.product(*Y_range_list)):
                p_Y_X_max += "p(%s | %s) = %0.6f\n" %(y, x, v_max_i[j])
            p_Y_X_max += "\n"

        print(p_Y_X_max)
    else:
        return (omegaG_pi, v_max)
