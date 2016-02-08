# StochDynamicProgramming



**WARNING:** *This package is currently in development, and is not operationnal yet.*
Every feedback would be welcomed.


[![Build Status](https://travis-ci.org/frapac/StochDynamicProgramming.svg?branch=master)](https://travis-ci.org/frapac/StochDynamicProgramming)
[![codecov.io](https://codecov.io/github/frapac/StochDynamicProgramming.jl/coverage.svg?branch=master)](https://codecov.io/github/frapac/StochDynamicProgramming.jl?branch=master)


This package implements the *Stochastic Dual Dynamic Programming* (SDDP) algorithm. It is built upon [JuMP](https://github.com/JuliaOpt/JuMP.jl)

It is currently under development.

We will shortly give some applications of this algorithms with the following examples:

- Dams valley management

- Unit-commitment

- Newsvendor testcase


IPython Notebooks will be provided to explain how this package work.


## Build documentation

The documentation is built with Sphinx, so ensure that this package is installed:

```bash
sudo apt-get install python-sphinx

```

To build the documentation:

```bash
cd doc
make html

```
