# StochDynamicProgramming



**WARNING:** *This package is currently in development, and is not operationnal yet.*


[![Build Status](https://travis-ci.org/leclere/StochDynamicProgramming.jl.svg?branch=master)](https://travis-ci.org/leclere/StochDynamicProgramming.jl)
[![codecov.io](https://codecov.io/github/leclere/StochDynamicProgramming.jl/coverage.svg?branch=master)](https://codecov.io/github/leclere/StochDynamicProgramming.jl?branch=master)


This is a Julia package for optimizing controlled stochastic dynamic system (in discrete time). It offers three methods of resolution :

- Extensive formulation
- Stochastic Dynamic Programming.
- *Stochastic Dual Dynamic Programming* (SDDP) algorithm. 

It is built upon [JuMP](https://github.com/JuliaOpt/JuMP.jl)

## What problem can we solve with this package ?

- Stage-wise independent discrete noise
- Linear dynamics
- Linear or convex piecewise linear cost
Extension to non-linear formulation are under development.

## Why Extensive formulation ?

An extensive formulation approach consists in representing the problem in a linear
problem solved by an external linear solver. Complexity is exponential in number of stages. 

## Why Stochastic Dynamic Programming ?

Dynamic Programming is a standard tool to solve stochastic optimal control problem with
independent noise. The method require discretisation of the state space, and is exponential
in the dimension of the state space.

## Why SDDP?

SDDP is a dynamic programming algorithm relying on cutting planes. The algorithm require convexity
of the value function but does not discretize the state space. The complexity is linear in the
number of stage, and can accomodate higher dimension state than standard dynamic programming. 
The algorithm return exact lower bound and estimated upper bound as well as approximate optimal
control strategies.



## Installation

```bash
Pkg.clone("https://github.com/leclere/StochDynamicProgramming.jl.git")

```

## Usage

IJulia Notebooks will be provided to explain how this package work.
A first example on a two dams valley [here.] (http://nbviewer.jupyter.org/github/leclere/StochDynamicProgramming.jl/blob/master/notebooks/damsvalley.ipynb)


## Documentation

The documentation is built with Sphinx, so ensure that this package is installed:

```bash
sudo apt-get install python-sphinx

```

To build the documentation:

```bash
cd doc
make html

```
