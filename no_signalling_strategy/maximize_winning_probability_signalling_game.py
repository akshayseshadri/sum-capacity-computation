import numpy as np
import scipy as sp
import cvxpy as cp
import itertools

from maximize_winning_probability_no_signalling_strategy import maximize_winning_prob_nonlocal_game

"""
    Finds the maximum winning probability of the signalling game using no-signalling (NS) strategies.

    Author: Akshay Seshadri
"""

def maximize_winning_prob_signalling_game(X1, X2, pi, print_result = False):
    """
        Finds the maximum winning probability for the signalling game using no-signalling strategies.

        In the signalling game, Alice and Bob receive questions from X_1 and X_2, respectively.
        They win the game if they infer the question received by the other person.
        That is, Y_1 = X_2 and Y_2 = X_1, and W = {((x_1, x_2), (y_1, y_2)) \in (X_1 x X_2) x (Y_1 x Y_2) | y_1 = x_2, y_2 = x_1}.

        args:
        - X1: number of questions to be given to Alice
        - X2: number of questions to be given to Bob
        - pi: element of the probability simplex in R^{|X_1 x X_2|}, with elements order as per itertools.product(X_1, ..., X_2)

        - print_result: print the maximum winning probability and the optimal strategy in a human-readable format
    """
    # winning condition
    W = [(x, x[::-1]) for x in itertools.product(*[range(X1), range(X2)])]

    if print_result:
        maximize_winning_prob_nonlocal_game(X_list = [X1, X2], Y_list = [X2, X1], W = W, pi = pi, print_result = True)
    else:
        omegaG_pi, v_max = maximize_winning_prob_nonlocal_game(X_list = [X1, X2], Y_list = [X2, X1], W = W, pi = pi, print_result = False)

        return (omegaG_pi, v_max)

def compare_maximum_uniform_winning_prob_signalling_game(X1_list, X2_list, print_result = True):
    """
        Finds the maximum winning probability for the signalling game using no-signalling strategies when the questions are drawn uniformly.

        The result is compared with the analytical expression
        omega_U(G) = 1/max(|X_1|, |X_2|)

        In the signalling game, Alice and Bob receive questions from X_1 and X_2, respectively.
        They win the game if they infer the question received by the other person.
        That is, Y_1 = X_2 and Y_2 = X_1, and W = {((x_1, x_2), (y_1, y_2)) \in (X_1 x X_2) x (Y_1 x Y_2) | y_1 = x_2, y_2 = x_1}.

        args:
        - X1: list of number of questions to be given to Alice
        - X2: list of number of questions to be given to Bob
        - print_result: print the numerically computed maximum winning probability, the analytical expression for maximum winning probability,
                        and the difference between the two
    """
    # number of question sets for Alice
    N1 = len(X1_list)
    # number of question sets for Bob
    N2 = len(X2_list)

    # numerically computed maximum uniform winning probability
    max_uniform_winning_prob_list = np.zeros((N1, N2))
    # analytical expression for the maximum uniform winning probability
    analytical_max_uniform_winning_prob_list = np.zeros((N1, N2))

    for (i, j) in itertools.product(range(N1), range(N2)):
        # number of questions for Alice & BOb
        X1, X2 = X1_list[i], X2_list[j]
        # winning condition
        W = [(x, x[::-1]) for x in itertools.product(*[range(X1), range(X2)])]
        # uniform probability distribution on the set of questions
        pi_U = np.ones(X1*X2) / (X1*X2)

        max_uniform_winning_prob_list[i, j] = maximize_winning_prob_nonlocal_game(X_list = [X1, X2], Y_list = [X2, X1], W = W, pi = pi_U, print_result = False)[0]
        analytical_max_uniform_winning_prob_list[i, j] = 1/max(X1, X2)

    if print_result:
        print("Maximize uniform winning probability computed numerically:")
        print(max_uniform_winning_prob_list)
        print("Maximize uniform winning probability from analytical expression:")
        print(analytical_max_uniform_winning_prob_list)
        print("Absolute value of difference between the two probabilities")
        print(np.abs(max_uniform_winning_prob_list - analytical_max_uniform_winning_prob_list))
        print("Maximum of absolute value of difference between the two probabilities")
        print(np.max(np.abs(max_uniform_winning_prob_list - analytical_max_uniform_winning_prob_list)))
    else:
        return (max_uniform_winning_prob_list, analytical_max_uniform_winning_prob_list)
