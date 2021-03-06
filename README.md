# StochDynamicProgramming


| **Documentation** | **Build Status** | **Social** |
|:-----------------:|:----------------:|:----------:|
| | [![Build Status][build-img]][build-url] | [![Gitter][gitter-img]][gitter-url] |
| [![][docs-stable-img]][docs-stable-url] |  [![Codecov branch][codecov-img]][codecov-url] | [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Discourse_logo.png/799px-Discourse_logo.png" width="64">][discourse-url] |

This is a Julia package for optimizing controlled stochastic dynamic system,
in discrete time. It offers three methods of resolution :

- *Stochastic Dual Dynamic Programming* (SDDP) algorithm.
- *Extensive formulation*.
- *Stochastic Dynamic Programming*.

It is built on top of [JuMP](https://github.com/JuliaOpt/JuMP.jl).

StochDynamicProgramming asks the user to provide explicit the cost `c(t, x, u, w)` and
dynamics `f(t, x, u, w)` functions. Also, the package was developed back
in 2016, and some parts of its API are not idiomatic, in a Julia sense.
For other implementations of the SDDP algorithm in Julia, we advise to
have a look at these two packages:

* [SDDP.jl](https://github.com/odow/SDDP.jl)
* [StructDualDynProg.jl](https://github.com/JuliaStochOpt/StructDualDynProg.jl)



## What problems solves this package ?

StochDynamicProgramming targets problems with

- Stage-wise independent discrete noise
- Linear dynamics
- Linear or convex piecewise linear costs

Extension to non-linear formulation are under development.


### Why SDDP?

SDDP is a dynamic programming algorithm relying on cutting planes. The algorithm requires convexity
of the value function but does not discretize the state space. The complexity is linear in the
number of stage, and can accomodate higher dimension state spaces than standard dynamic programming.
The algorithm returns exact lower bound and estimated upper bound as well as approximate optimal
control strategies.

### Why Extensive formulation ?

An extensive formulation approach consists in representing the stochastic problem as a deterministic
one and then calling a standard deterministic solver. It is mainly usable in a linear
setting. Computational complexity is exponential in the number of stages.

### Why Stochastic Dynamic Programming ?

Dynamic Programming is a standard tool to solve stochastic optimal control problem with
independent noise. The method requires discretizing the state space, and its
complexity is exponential in the dimension of the state space.


## Installation

StochDynamicProgramming is a registered Julia package.
To install the package, open Julia and enter

```julia
julia> ]
pkg> add StochDynamicProgramming

```


## Usage

IJulia Notebooks are provided to explain how this package works.
A first example on a two dams valley [here](http://nbviewer.jupyter.org/github/leclere/StochDP-notebooks/blob/master/notebooks/damsvalley.ipynb).


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

## License

Released under Mozilla Public License (see LICENSE.md for further details).



[build-img]: https://travis-ci.org/JuliaStochOpt/StochDynamicProgramming.jl.svg?branch=master
[build-url]: https://travis-ci.org/JuliaStochOpt/StochDynamicProgramming.jl
[codecov-img]: https://codecov.io/github/JuliaStochOpt/StochDynamicProgramming.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/github/JuliaStochOpt/StochDynamicProgramming.jl?branch=master
[gitter-url]: https://gitter.im/JuliaOpt/StochasticDualDynamicProgramming.jl
[gitter-img]: https://badges.gitter.im/JuliaStochOpt/StochasticDualDynamicProgramming.jl.svg
[discourse-url]: https://discourse.julialang.org/c/domain/opt
[JuMP]: https://github.com/JuliaOpt/JuMP.jl
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://stochdynamicprogramming.readthedocs.io/en/latest/

