## Scalable Combinatorial Bayesian Optimization with Tractable Statistical models


This repository contains the source code for the paper "[Scalable Combinatorial Bayesian Optimization with Tractable Statistical models](https://arxiv.org/abs/2008.08177)". In this paper, we employ the submodular relaxation based optimization approach for Binary Quadratic Programs proposed in [Ito and Fujimaki](https://papers.nips.cc/paper/6301-large-scale-price-optimization-via-network-flow) to improve the computational-efficiency and accuracy of solving acquisition function optimization (AFO) problems for [BOCS](https://arxiv.org/abs/1806.08838) which is a state-of-the-art method for black-box optimization of combinatorial input spaces. The repository builds upon the [source code](https://github.com/baptistar/BOCS) provided by the BOCS authors. 



### Requirements/Installation
The code is implemented in Python and requires the following libraries for different acquisition function optimizers:
1. [graph-tool](https://graph-tool.skewed.de/) for submodular relaxation approach
2. [cvxpy](https://www.cvxpy.org/) and [cvxopt](https://cvxopt.org/) for semi-definite relaxation approach


### Benchmarks

There are 3 benchmarks provided in the files named example_contamination.py, example_ising.py, and example_bqp.py. They can be run on command line via python 'name_of_the_file'. Most of the code is self-contained and documented. Please e-mail at aryan.deshwal@wsu.edu if you have any issue. 
