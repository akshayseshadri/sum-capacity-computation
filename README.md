# Computing the sum capacity of two-sender multiple access channels

We present a algorithms for performing the following tasks:

1. Optimizing Lipschitz-like functions over the standard simplex.

2. Computing the sum capacity of two-sender multiple access channels (MACs).

3. Computing the maximum winning probability of N-player nonlocal games using no-signalling (NS) strategies.

## Optimizing Lipschitz-like functions
We present two different algorithms for maximizing Lipschitz-like functions over the standard simplex:

- A modified Piyavskii-Shubert algorithm that computes the maximum over an interval by finding successively better upper bounds on the objective function.\
  [see [lipschitzlike_optimization/maximize_lipschitzlike_function_interval.py](lipschitzlike_optimization/maximize_lipschitzlike_function_interval.py)]

  The modified Piyavskii-Shubert algorithm is generalized to handle optimization over the standard simplex by constructing a dense curve that fill the simplex.\
  [see [lipschitzlike_optimization/maximize_lipschitzlike_function_standard_simplex_dense_curve.py](lipschitzlike_optimization/maximize_lipschitzlike_function_standard_simplex_dense_curve.py),\
       [lipschitzlike_optimization/fill_simplex.py](lipschitzlike_optimization/fill_simplex.py)]

- A grid search based algorithm that computes the maximum over a grid on the standard simplex.\
  [see [lipschitzlike_optimization/maximize_lipschitzlike_function_standard_simplex_grid_search.py](lipschitzlike_optimization/maximize_lipschitzlike_function_standard_simplex_grid_search.py)]

  We present an algorithm to efficiently query the elements of the grid without constructing the grid.\
  [see the function `get_grid_point` in [lipschitzlike_optimization/fill_simplex.py](lipschitzlike_optimization/fill_simplex.py)]\
  This function can be used to parallelize the grid search (note that parallelization has not been implemented in grid serch).

  Check out the examples in [lipschitzlike_optimization/lipschitzlike_maximization_examples.py](lipschitzlike_optimization/lipschitzlike_maximization_examples.py).

## Sum capacity computation
Using the algorithms for optimizing Lipschitz-like functions, we present two algorithms (based on dense curves and grid search) for computing the sum capacity of two-sender MACs to a given precision.\
[see [compute_sum_capacity/compute_sum_capacity_twosenderMAC.py](compute_sum_capacity/compute_sum_capacity_twosenderMAC.py)]

Note that we allow for the possibility to specify `max_iter`, which is the maximum number of iterations for which the algorithm will run.\
When using the algorithm based on dense curve, one can use the value of the bounding function at `max_iter` iteration as an upper bound on the sum capacity.

Check out some examples of sum capacity computation using our algorithms in [compute_sum_capacity/sum_capacity_computation_twosenderMAC_examples.py](compute_sum_capacity/sum_capacity_computation_twosenderMAC_examples.py).

## Maximum winning probability of nonlocal games using NS strategies
We present an algorithm to compute the maximum winning probability of any given N-player nonlocal game using no-signalling strategies when the questions are drawn as per some specified probability distribution.\
[see [no_signalling_strategy/maximize_winning_probability_no_signalling_strategy.py](no_signalling_strategy/maximize_winning_probability_no_signalling_strategy.py)]

This computation is done by appropriately encoding the constraints placed by the no-signalling requirement, and solving the resulting linear program.

Owing to such an approach, one can use the same code to maximize any concave function of no-signalling distributions
by appropriately modifying the objective function in the code.

As an application of our code, we compute the maximum winning probability of the signalling game using NS strategies when the questions are drawn uniformly at random.

# Dependencies and Installation
The code has been implemented in Python 3 (3.10.2).

The following modules are necessary to run the code: numpy (1.22.0), scipy (1.7.3), and cvxpy (1.1.18).

We specify the version number using which we tested the code in brackets.

No installation is necessary and the code can be run as is.

# License
We release the code under [MIT license](LICENSE).

# Reference
Please consider citing our work TODO if you are using our code in your work.
