.. _quickstart:

====================
Step-by-step example
====================

This page gives a short introduction to the interface of this package. It explains the resolution with SDDP of a classical example: the management of a dam over one year with random inflow.

Use case
========
In the following, :math:`x_t` will denote the state and :math:`u_t` the control at time :math:`t`.
We will consider a dam, whose dynamic is:

.. math::
   x_{t+1} = x_t - u_t + w_t

At time :math:`t`, we have a random inflow :math:`w_t` and we choose to turbine a quantity :math:`u_t` of water.

The turbined water is used to produce electricity, which is being sold at a price :math:`c_t`. At time :math:`t` we gain:

.. math::
    C(x_t, u_t, w_t) = c_t \times u_t

We want to minimize the following criterion:

.. math::
    J = \underset{x, u}{\min} \sum_{t=0}^{T-1} C(x_t, u_t, w_t)

We will assume that both states and controls are bounded:

.. math::
    x_t \in [0, 100], \qquad u_t \in [0, 7]


Problem definition in Julia
===========================

We will consider 52 time steps as we want to find optimal value functions for one year::

    N_STAGES = 52


and we consider the following initial position::

    X0 = [50, 50]


Dynamic
^^^^^^^

We write the dynamic::

    function dynamic(t, x, u, xi)
        return [x[1] + u[1] - xi[1]]
    end


Cost
^^^^

we store evolution of costs :math:`c_t` in an array `COSTS`, and we define the cost function::

    function cost_t(t, x, u, w)
        return COSTS[t] * u[1]
    end

Noises
^^^^^^

Noises are defined in an array of Noiselaw. This type defines a discrete probability.


For instance, if we want to define a uniform probability with size :math:`N= 10`, such that:

.. math::
    \mathbb{P} \left(X_i = i \right) = \dfrac{1}{N} \qquad \forall i \in 1 .. N

we write::

    N = 10
    proba = 1/N*ones(N) # uniform probabilities
    xi_support = collect(linspace(1,N,N))
    xi_law = NoiseLaw(xi_support, proba)


Thus, we could define a different probability laws for each time :math:`t`. Here, we suppose that the probability is constant over time, so we could build the following vector::

    xi_laws = NoiseLaw[xi_law for t in 1:N_STAGES-1]


Bounds
^^^^^^

We could add bounds over the state and the control::

    s_bounds = [(0, 100)]
    u_bounds = [(0, 7)]


Problem definition
^^^^^^^^^^^^^^^^^^

As our problem is purely linear, we could instantiate::

    spmodel = LinearDynamicLinearCostSPmodel(N_STAGES,u_bounds,X0,cost_t,dynamic,xi_laws)


Solver
^^^^^^
We define a SDDP solver for our problem.

First, we need to use a LP solver::

    using Clp
    SOLVER = ClpSolver()

Clp is automatically installed during package installation. To install the solver on your machine, refer to the JuMP_ documentation.

Once the solver installed, we could define the parameters of the SDDP algorithm::

    forwardpassnumber = 2 # number of forward pass
    sensibility = 0. # admissible gap between upper and lower bound
    max_iter = 20  # maximum number of iterations

    paramSDDP = SDDPparameters(SOLVER, forwardpassnumber, sensibility, max_iter)


Now, we could compute Bellman values::

    V, pbs = solve_SDDP(spmodel, paramSDDP, 10) # display information every 10 iterations

:code:`V` is an array storing the value functions, and :code:`pbs` a vector of JuMP.Model storing each value functions as a linear problem.

We could estimate the lower bound given by :code:`V` with the function::

    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, V)


Find optimal control over given scenarios
=========================================

Once Bellman functions are computed, we could control our system over assessments scenarios.

We build 1000 scenarios according to the laws stored in :code:`xi_laws`::

    scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws,1000)

And we could compute 1000 simulations over these scenarios::

    costsddp, stocks = forward_simulations(spmodel, paramSDDP, V, pbs, scenarios)

:code:`costsddp` returns the value of costs along each scenario, and :code:`stocks` the evolution of each stock along time.

.. _JuMP: http://jump.readthedocs.io/en/latest/installation.html#coin-or-clp-and-cbc

