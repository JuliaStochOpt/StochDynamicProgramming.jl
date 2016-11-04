
========
SDDP: Advanced functions
========

This page gives an overview of the functions implemented in the package.

In the following, :code:`model` will design a :code:`SPModel` storing the
definition of a stochastic problem, and :code:`param` a SDDPparameters instance
which stores the SDDP's parameters. See quickstart_ for more
information about these two objects.

Work with PolyhedralFunction
============================

Get Bellman value at a given point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To estimate the Bellman value at a given position :code:`xt` with a :code:`PolyhedralFunction` :code:`Vt` ::

    vx = get_bellman_value(model, param, t, Vt, xt)

This is a lower bound of the true expectation cost.

Get optimal control at a given point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get optimal control at a given point :code:`xt` and for a given alea :code:`xi`::

    get_control(model, param, lpproblem, t, xt, xi)

where :code:`lpproblem` is the linear problem storing the evaluation of
Bellman function at time :math:`t`.



Save and load pre-computed cuts
===============================

Assume that we have computed Bellman functions with SDDP. These functions are
stored in a vector of :code:`PolyhedralFunctions` denoted :code:`V`

These functions can be stored in a text file with the command::

    StochDynamicProgramming.dump_polyhedral_functions("yourfile.dat", V)

And then be loaded with the function::

    Vdump = StochDynamicProgramming.read_polyhedral_functions("yourfile.dat")



Build LP Models with PolyhedralFunctions
=======================================

We can build a vector of :code:`JuMP.Model` with a vector of
:code:`PolyhedralFunction` to perform simulation. For this, use the function::

    problems = StochDynamicProgramming.hotstart_SDDP(model, param, V)


SDDP hotstart
=============

If cuts are already available, we can hotstart SDDP while overloading the function :code:`solve_SDDP`::

    V, pbs = solve_SDDP(model, params, V, 0)


Cuts pruning
============

The more SDDP run, the more cuts you need to store. It is sometimes useful to
delete cuts which are useless for the computation of the approximated Bellman functions.


To clean a single :code:`PolyhedralFunction` :code:`Vt`::

    Vt = exact_prune_cuts(model, params, Vt)

To clean a vector of :code:`PolyhedralFunction` :code:`V`::

    prune_cuts!(model, params, V)


.. _quickstart: quickstart.html
