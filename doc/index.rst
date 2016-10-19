.. StochDynamicProgramming documentation master file, created by
   sphinx-quickstart on Mon Nov 30 16:56:49 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================
StochDynamicProgramming
=======================

This package implements the Stochastic Dual Dynamic Programming (SDDP)
algorithm with Julia. It relies upon MathProgBase_ and JuMP_, and is compatible
with both Julia v0.4 and v0.5.

A complete overview of this algorithm could be found here_.

Description of SDDP
^^^^^^^^^^^^^^^^^^^

At the moment:

- you can fix linear or quadratic time :math:`t` cost function (then convex piecewise linear)
- you can fix linear dynamic (with a physical state :math:`x` and a control :math:`u`)
- the scenarios are created by assuming that the noise :math:`\xi_t` is independent in time, each given by a table (value, probability)


Then it is standard SDDP :

- fixed number of forward passes
- iterative backward passes
- no clearing of cuts
- stopping after a given number of iterations / computing time

Once solved the SDDP model should be able to :

- give the lower bound on the cost and its evolution along iterations
- simulate trajectories to evaluate expected cost
- give the optimal control given current time state and noise


Supported features
^^^^^^^^^^^^^^^^^^

.. .. table:
.. ======  =========== ===============
.. Solver  Is working? Quadratic costs
.. ======  =========== ===============
.. Linear Cost     Yes         No
.. Quadratic Cost   Yes         Yes
.. Integer controls
.. Quadratic regularization
.. Cuts pruning


Ongoing developments
^^^^^^^^^^^^^^^^^^^^

We have a lot of ideas to implement further :

- spatial construction (i.e : defining stock one by one and then their interactions)
- noise as AR (eventually with fitting on historic datas)
- convex solvers
- refined stopping rules



Contents:
=========

.. toctree::
   install
   quickstart
   sddp_api
   tutorial
   quickstart_sdp
   install_windows

.. _MathProgBase: http://mathprogbasejl.readthedocs.org/en/latest/
.. _here: http://www2.isye.gatech.edu/people/faculty/Alex_Shapiro/ONS-FR.pdf
.. _JuMP: http://jump.readthedocs.org/en/latest/

