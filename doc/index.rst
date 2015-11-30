.. StochDynamicProgramming documentation master file, created by
   sphinx-quickstart on Mon Nov 30 16:56:49 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

StochDynamicProgramming Index
=============================

This package implements the Stochastic Dual Dynamic Programming (SDDP) algorithm with Julia. It relies upon MathProgBase_.

A complete overview of this algorithm could be found here_.

At the moment the plan is to create a type such that :

- you can fix linear time :math:`t` cost function (then convex piecewise linear)
- you can fix linear dynamic (with a physical state :math:`x` and a control :math:`u`)
- the scenarios are created by assuming that the noise :math:`\xi_t` is independent in time, each given by a table (value, probability)


Then it is standard SDDP :

- fixed number of forward passes
- iterative backward passes
- no clearing of cuts
- stopping after a given number of iterations / computing time

Once solved the SDDP model should be able to :

- give the lower bound on the cost
- simulate trajectories to evaluate expected cost
- give the optimal control given current time state and noise


We have a lot of ideas to implement further :

- spatial construction (i.e : defining stock one by one and then their interactions)
- noise as AR (eventually with fitting on historic datas)
- convex solvers
- refined stopping rules
- cut pruning
- parralellization



Contents:

.. toctree::
   :maxdepth: 2

.. _MathProgBase: http://mathprogbasejl.readthedocs.org/en/latest/
.. _here: http://www2.isye.gatech.edu/people/faculty/Alex_Shapiro/ONS-FR.pdf

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

