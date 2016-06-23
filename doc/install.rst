.. _install:

==================
Installation guide
==================


StochDynamicProgramming installation
------------------------------------

To install StochDynamicProgramming::

    julia> Pkg.add("StochDynamicProgramming")


Once the package is installed, you can import it in the REPL::

    julia> using StochDynamicProgramming


Install a linear programming solver
-----------------------------------

SDDP need a linear programming (Clp, Gurobi, CPLEX, etc.) solver to run. To install Clp::

    julia> Pkg.add("Clp")

Refer to the documentation of JuMP_ to get another solver and interface it with SDDP.



The following solvers have been tested:

.. table:
======  ===========
Solver  Is working?
======  ===========
Clp     Yes
CPLEX   Yes
Gurobi  Yes
GLPK    **No**
======  ===========

Run Unit-Tests
--------------
To run unit-tests::

   $ julia test/runtests.jl


.. _JuMP: http://jump.readthedocs.org/en/latest/installation.html#getting-solvers

