import numpy as np
import scipy as sp
import warnings

from scipy import linalg
from scipy.special import comb

"""
    Constructs a dense curve to fill the standard simplex.

    Also contains an algorithm to efficiently obtain an element of the grid over the standard simplex,
    ordered as per Definition (9) of the article
    "On the separation of correlation-assisted sum capacities of multiple access channels"

    Author: Akshay Seshadri
"""

def get_grid_point(grid_ind, N, d, normalized = True):
    """
        Compute the grid point given the index grid_ind of the grid

        \Delta_{d, N} = {(n_1, ..., n_d)/N | n_1, ..., n_d \in \mathbb{N}, \sum_{i = 1}^d n_i = N}

        ordered as per Definition (9) of the article
        "On the separation of correlation-assisted sum capacities of multiple access channels".

        The grid point is computed without constructing the grid.

        Args:
            - grid_ind   : index of the desired grid point (starting with the index zero) of the grid \Delta_{d, N}
                           following the ordering specified above
            - N          : parameter specifying the distance between consecutive grid points as per above ordering
                           (distance between consecutive grid points is 2/N in l1 norm)
            - d          : dimension of the Euclidean space containing the grid
            - normalized : specifies whether the sum entries of the grid point is normalized to 1

        Output:
            grid point determined by the index grid_ind
    """
    # since we will be comparing number of points until we reach grid index,
    # and we follow zero-index convention for grid_ind, add one to grid index
    grid_ind += 1

    # keep track of whether the location in the current iteration is in the front or back side of the list
    # front corresponds to 1 and back corresponds to -1
    current_dir = 1

    # index in front and back directions of the list that have been filled until now
    front = 0
    back = 0

    # current index in the list
    ind = 0

    # sum of coordinates, which shrinks as we traverse through the list and infer grid coordinates
    coord_sum = N

    # analytical formula for number of grid points given the number 'n' to which the coordinates sum,
    # assuming that there are a total of i coordinates
    # the term 1 appears to account for the connection between from coordinate i to the adjacent point in coordinate i - 1
    def grid_size(n, i):
        if i > 0:
            return comb(n + i - 1, n, 'exact')
        else:
            return 1

    if grid_ind > grid_size(N, d):
        raise ValueError("grid_ind must be less than comb(N + d - 1, d - 1) = %d" %(grid_size(N, d),))

    # warn of potential overflow error for large grid sizes
    if grid_size(N, d) > 1e18:
        warnings.warn("The grid size is too large (greater than 10^18) and may result in overflow errors.")

    # grid size traversed until previous dimension
    traversed_size = 0

    # iteratively compute the grid point corresponding to grid_ind
    # the grid point is parametrized by the integer tuple (n_{d - 1}, ..., n_1)
    grid_ind_point = np.zeros(d)

    # index of grid_ind_point where the point needs to be filled
    ind = 0

    for i in range(d, 0, -1):
        # compute the location for the ith index corresponding to where grid_ind falls
        # n_i is the sum of the remaining (uninferred) coordinates at the ith iteration
        # at the ith iteration, we want n_{i - 1} such that
        # grid index falls between (n_i - n_{i - 1}, ..., n_2 - n_1, n_1) and next point
        # (the order of the point may be reversed, depending on whether n_i is odd or even)
        grid_size_list = np.cumsum(np.array([grid_size(n_i_1, i - 1) for n_i_1 in range(0, coord_sum + 1)]))

        # find the two blocks (determined by n_{i - 1}) between which the point x(theta) lies
        n_i_1 = np.where(traversed_size + grid_size_list <= grid_ind)[0][-1]

        if traversed_size + grid_size_list[n_i_1] < grid_ind:
            n_i_1 += 1

        # fill the currently known location of the grid point
        grid_ind_point[ind] = coord_sum - n_i_1

        # update size of grid traversed
        if n_i_1 > 0:
            traversed_size += grid_size_list[n_i_1 - 1]
        else:
            if current_dir == 1:
                grid_ind_point[front: d - back] = [coord_sum] + [0] * (d - back - front - 1)
            else:
                grid_ind_point[front + 1: d - back + 1] = [0] * (d - back - front - 1) + [coord_sum]
            break

        # infer the location for the next point
        # direction of traversal depends on whether n_{i - 1} is even or odd
        # and on whether coord_sum is even or odd
        # coord_sum odd, n_{i - 1} odd  -> forward traversal,  denoted by 1
        # coord_sum odd, n_{i - 1} even -> backward traversal, denoted by -1
        # coord_sum even, n_{i - 1} even  -> forward traversal,  denoted by 1
        # coord_sum even, n_{i - 1} odd -> backward traversal, denoted by -1
        traversal_dir = (2*(coord_sum%2) - 1) * (2*(n_i_1%2) - 1)
        if current_dir == 1:
            if traversal_dir == 1:
                front += 1
                ind = front
                current_dir *= 1
            else:
                back += 1
                ind = d - back
                current_dir *= -1
        else:
            if traversal_dir == 1:
                back += 1
                ind = d - back
                current_dir *= 1
            else:
                front += 1
                ind = front
                current_dir *= -1

        # the sum of the remaining coords is equal to n_{i - 1}
        coord_sum = n_i_1

    # if specified, normalize the grid point so that it lies on the simplex
    if normalized:
        return grid_ind_point / N
    else:
        return grid_ind_point

def obtain_grid_curve_point(theta, N, d):
    """
        Construct a curve that connects all points of the grid

            \Delta(N, d) = {x \in \Delta_d | N x \in \mathbb{N}^d}

        of the (d - 1)-dimensional standard simplex

            \Delta_d = {x \in R^d | x >= 0, \sum(x) = 1}.

        The curve is defined to preserve the length in the sense that

            ||x(theta_1) - x(theta_2)||_1 \leq \min{|theta_1 - theta_2|, 2}

        for all theta_1, theta_2, and

            ||x(theta_1) - x(theta_2)||_1 = |theta_1 - theta_2|

        between two consecutive grid points connected by the curve.

        Args:
            - theta: parameter characterizing the curve; lies in the interval [0, L(d)], 
                     where L(N, d) denotes the length of the curve required to connect
                     the grid \Delta(N, d)
            - N: number to which the elements of the grid sum to
            - d: dimension of the space containg the simplex \Delta_d

        Output:
            - x: a point in the simplex obtained using the curve for the parameter 'theta'
    """
    # first, find the index in the full list of (ordered) grid points such that
    # the point x(theta) on the curve lies between grid[index] and grid[index + 1]
    # since the neighbouring grid points in the ordered grid are equidistant (with distance of 2/N),
    # the index can be directly inferred from theta
    grid_ind = np.floor(theta * N/2)

    # handle the edge cases
    if np.abs(grid_ind) < 1e-5:
        return get_grid_point(grid_ind, N, d)
    if np.abs(grid_ind - comb(N + d - 1, d - 1, 'exact')) < 1e-5:
        return get_grid_point(grid_ind - 1, N, d)

    # compute the grid point corresponding to grid_ind and the next point
    grid_point = get_grid_point(grid_ind, N, d)
    grid_point_next = get_grid_point(grid_ind + 1, N, d)

    # obtain the point x(theta) by taking appropriate convex combination of the two grid points x_1, x_2
    # ||t x_1 + (1 - t) x_2 - x_1||_1 = (1 - t) ||x_1 - x_2||_1 = (1 - t) * 2/N = theta - grid_ind * 2/N
    t = 1 + grid_ind - theta * N/2

    # obtain x(theta)
    x_theta = t * grid_point + (1 - t) * grid_point_next

    return x_theta
