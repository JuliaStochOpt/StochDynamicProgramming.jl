.. _quickstart:

====================
SDDP: Step-by-step example
====================

This page gives a short introduction to the interface of this package. It
explains the resolution with SDDP of a classical example: the management of a
dam over one year with random inflow.

Use case
========
In the following, :math:`x_t` denotes the state and :math:`u_t` the control at time :math:`t`.
We consider a dam, whose dynamic is:

.. math::
   x_{t+1} = x_t - u_t + w_t

At time :math:`t`, we have a random inflow :math:`w_t` and we choose to turbine
a quantity :math:`u_t` of water.

The turbined water is used to produce electricity, which is being sold at a
price :math:`c_t`. At time :math:`t` we gain:

.. math::
    C(x_t, u_t, w_t) = c_t \times u_t

We want to minimize the following criterion:

.. math::
    J = \underset{x, u}{\min} \mathbb{E} \;\left[ \sum_{t=0}^{T-1} C(x_t, u_t, w_t) \right]

We assume that both states and controls are bounded:

.. math::
    x_t \in [0, 100], \qquad u_t \in [0, 7]


Problem definition in Julia
===========================

We consider 52 time steps as we want to find optimal value functions every week
during one year::

    N_STAGES = 52


and we consider the following initial position::

    X0 = [50]

Note that `X0` is a vector.

Dynamic
^^^^^^^

We write the dynamic (which return a vector)::

    function dynamic(t, x, u, xi)
        return [x[1] + u[1] - xi[1]]
    end


Cost
^^^^

We store evolution of costs :math:`c_t` in an array `COSTS`, and we define
the cost function (which return a float)::

    function cost_t(t, x, u, w)
        return COSTS[t] * u[1]
    end

Noises
^^^^^^

Noises are defined in an array of `Noiselaw`. This type defines a discrete probability.


For instance, if we want to define a uniform probability with size :math:`N= 10`, such that:

.. math::
    \mathbb{P} \left(X_i = i \right) = \dfrac{1}{N} \qquad \forall i \in 1 .. N

we write::

    N = 10
    proba = 1/N*ones(N) # uniform probabilities
    xi_support = collect(linspace(1, N, N))
    xi_law = NoiseLaw(xi_support, proba)


Thus, we could define a different probability laws for each time :math:`t`. Here, we suppose that the probability is constant over time, so we could build the following vector::

    xi_laws = NoiseLaw[xi_law for t in 1:N_STAGES-1]


Bounds
^^^^^^

Both state and control are bounded:

    s_bounds = [(0, 100)]
    u_bounds = [(0, 7)]


Problem definition
^^^^^^^^^^^^^^^^^^

As our problem is purely linear, we instantiate::

    spmodel = LinearSPModel(N_STAGES, u_bounds, X0, cost_t, dynamic, xi_laws)

We add the state bounds to the model afterward::

    set_state_bounds(spmodel, s_bounds)


Solver
^^^^^^
We define a SDDP solver for our problem.

First, we need to use a LP solver::

    using Clp
    SOLVER = ClpSolver()

Clp is automatically installed during packages' installation. To install
different solvers on your machine, refer to the JuMP_ documentation.

Once the solver is installed, we define SDDP parameters::

    forwardpassnumber = 2 # number of forward pass
    gap = 0. # admissible gap between upper and lower bound
    max_iter = 20  # maximum number of iterations

    paramSDDP = SDDPparameters(SOLVER,
                               passnumber=forwardpassnumber,
                               gap=gap,
                               max_iterations=max_iter)


Now, we solve the problem by computing Bellman values::

    V, pbs, stats = solve_SDDP(spmodel, paramSDDP, 10) # display information every 10 iterations

:code:`V` is an array storing the value functions, and :code:`pbs` a vector of
JuMP.Model storing each value functions as a linear problem.
:code:`stats` is an object which stores a few informations about the convergence
of SDDP (execution time, evolution of upper and lower bounds, number of calls to
solver, etc.).

The exact lower bound is given by the function::

    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, V)


Find optimal controls
=====================

Once Bellman functions are computed, we can control our system over
assessments scenarios, without assuming knowledge of the future.

We build 1000 scenarios according to the laws stored in :code:`xi_laws`::

    scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws, 1000)

We compute 1000 simulations of the system over these scenarios::

    costsddp, stocks = forward_simulations(spmodel, paramSDDP, pbs, scenarios)

:code:`costsddp` returns the costs for each scenario, and :code:`stocks`
the evolution of stocks along time, for each scenario.

.. _JuMP: http://jump.readthedocs.io/en/latest/installation.html#coin-or-clp-and-cbc

