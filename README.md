# StochDynamicProgramming



**WARNING:** *This package is currently in development, and is not operationnal yet.*


[![Build Status](https://travis-ci.org/leclere/StochDynamicProgramming.jl.svg?branch=master)](https://travis-ci.org/leclere/StochDynamicProgramming.jl)
[![codecov.io](https://codecov.io/github/leclere/StochDynamicProgramming.jl/coverage.svg?branch=master)](https://codecov.io/github/leclere/StochDynamicProgramming.jl?branch=master)


This is a Julia implementation of the *Stochastic Dual Dynamic Programming* (SDDP) algorithm. It is built upon [JuMP](https://github.com/JuliaOpt/JuMP.jl)


## Why SDDP?

SDDP is an algorithm to solve multistage stochastic optimization problems.
It return bounds on the value of the optimization problem and approximation
of Bellman function that are used to derive an optimal solution.

These problems are modelled with:

- stage-wise independent discrete noise

- Linear dynamic

- Linear or piecewise linear cost


This algorithm could be applied to the following examples:

- Dams valley management

- Newsvendor testcase


The documentation will be soon updated to explain how this algorithm work.


## Installation

```bash
Pkg.clone("https://github.com/leclere/StochDynamicProgramming.jl.git")

```

## Usage

IJulia Notebooks will be provided to explain how this package work.
A first example on a two dams valley [here]. (http://nbviewer.jupyter.org/github/leclere/StochDynamicProgramming.jl/blob/master/notebooks/damsvalley.ipynb)


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
