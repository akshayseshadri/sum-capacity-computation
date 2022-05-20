import numpy as np
import scipy as sp
import bisect
import warnings

from scipy import stats, optimize

"""
    Maximization of Lipschitz-like functions over a closed interval using modified Piyavskii-Schubert algorithm.

    Author: Akshay Seshadri
"""

class Maximize_Lipschitzlike_Function_Interval():
    """
        A class that handles the maximization of Lipschitz-like functions over a closed interval using Piyavskii-Schubert algorithm in one dimension.
    """
    def __init__(self, f, beta, a, b, eps = 1e-1, max_iter = 1e3, quiet = False):
        """
            Initialize the variables necessary for performing the optimization.

            Args:
                - f: Lipschitz-like function from R to R
                - beta: function defining Lipschitz-like property
                - a: lower bound of the interval
                - b: upper bound of the interval
                - eps: precision to which the maximum is computed
                - max_iter: maximum number of global iterations to be allowed
                - quiet: suppress printing of output
        """
        ### initialize the variables needed for optimization
        # the objective function
        self.f = f

        # Lipschitz-like defining function
        self.beta = beta

        # start and end point of the interval, respectively
        if not a <= b:
            raise ValueError("'a' must be less than or equal to 'b'")

        self.a = a
        self.b = b

        # precision to which the optimum must be computed
        self.eps = eps

        # maximum number of iterations to be allowed
        self.max_iter = max_iter

        # specifies whether to print
        self.quiet = quiet

        ### initialize the lists needed to perform the optimization
        # list of maxima of the bounding function at each iteration
        # the maximum of F_bound defined with respect to q_0 is simply q = 1, so add that directly
        # the list is maintained in a sorted order
        self.q_max_list = [a, b]

        # store the values of f(q_i) to prevent repeated computation
        # the list is maintained in an order sorted as per q_max_list
        self.f_val_list = [f(a), f(b)]

    def get_bounding_function_maximum(self):
        """
            Given q_0, ..., q_k, define F_i(q) = f(q_i) + beta(|q - q_i|) for 0 <= i <= k

            The bounding function is defined as F_bound = min_{0 <= i <= k} F_i.

            The maximum of F_bound occurs at either the intersection point of consequtive F_i or at the boundary (a or b).
            The boundaries are not checked since q_0 = a and q_1 = b always.

            Computed as per Algorithm 2 (described in the notes).
            brent's method for root-finding is used (instead of bisection) for reasons of speed
        """
        # ensure that q_list and f_list are numpy arrays
        beta = self.beta
        q_list = self.q_max_list
        f_list = self.f_val_list

        # compute the intersection of F_i with F_{i + 1} for 0 <= i < k
        # this is done by finding a root of g_i = F_i - F_{i + 1}
        g = lambda q, count: f_list[count] - f_list[count + 1] + beta(np.abs(q - q_list[count])) - beta(np.abs(q - q_list[count + 1]))

        # number of points to iterate through
        k = len(q_list)

        # list of roots
        root_list = np.zeros(k - 1)

        # find the roots of g_i
        for count in range(k - 1):
            # find a root of g_i
            root_list[count] = sp.optimize.brentq(g, q_list[count], q_list[count + 1], args = (count,))

        # compute the bounding function value at each of the roots -- this amounts to computing F_i at the ith root
        F_bound_root_list = [f_list[count] + beta(np.abs(root - q_list[count])) for (count, root) in enumerate(root_list)]

        # find the index corresponding to the largest value of F_bound_root_list
        root_max_index = np.argmax(F_bound_root_list)
        root_max = root_list[root_max_index]
        F_bound_max = F_bound_root_list[root_max_index]

        return (root_max, F_bound_max)

    def maximize(self):
        """
            Computes the maximum of the Lipschitz-like function f, where f satisfies
                
                |f(x) - f(y)| <= beta(|x - y|)
        """
        # current iteration counter
        count = 0

        # bounding function maximum at the given count (time instant)
        F_bound_max = self.f_val_list[0] + self.beta(self.b)

        # current maximum value of the function
        f_max = self.f_val_list[-1]

        # break after maximum number of iterations is reached to avoid the  while loop from running ad infinitum
        while (F_bound_max - f_max > self.eps) and (count <= self.max_iter):
            # update the count
            count += 1

            # maximize the current bounding function
            q_max, F_bound_max = self.get_bounding_function_maximum()

            # update the current maximum list while mainting the sorting
            # find the location where q_max must be inserted to retain the sorting order
            loc = bisect.bisect_left(self.q_max_list, q_max)

            # insert the point at the sorted location
            self.q_max_list.insert(loc, q_max)

            # current value of maximum
            f_max = self.f(q_max)

            # update the list of f(q_i) values, using the location obtained from sorting q_max_list
            self.f_val_list.insert(loc, f_max)

            if count % 100 == 0 and not self.quiet:
                print("Iteration count:", count, end = "\r")

        if count > self.max_iter:
            warnings.warn("Modified Piyavskii-Schubert algorithm did not converge to the specified threshold within max_iter iterations. Consider increasing eps or max_iter (or both).")

        if not self.quiet:
            print("-------- Maximize_Lipschitzlike_Function_Interval ---------")
            print("Number of iterations:", count)
            print("Current bound", F_bound_max)
            print("Current error", F_bound_max - f_max)
            print("-------- Maximize_Lipschitzlike_Function_Interval ---------")

        return f_max
