.. _quickstart_sdp:

====================
SDP: Step-by-step example
====================

This page gives a short introduction to the interface of this package. It explains the resolution with Stochastic Dynamic Programming of a classical example: the management of a dam over one year with random inflow.

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

    X0 = [50]

Note that X0 is a vector.

Dynamic
^^^^^^^

We write the dynamic (which return a vector)::

    function dynamic(t, x, u, xi)
        return [x[1] + u[1] - xi[1]]
    end


Cost
^^^^

we store evolution of costs :math:`c_t` in an array `COSTS`, and we define the cost function (which return a float)::

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

We add bounds over the state and the control::

    s_bounds = [(0, 100)]
    u_bounds = [(0, 7)]


Problem definition
^^^^^^^^^^^^^^^^^^

We have two options to contruct a model that can be solved by the SDP algorithm.
We can instantiate a model that can be solved by SDDP as well::

    spmodel = LinearDynamicLinearCostSPmodel(N_STAGES,u_bounds,X0,cost_t,dynamic,xi_laws)

    set_state_bounds(spmodel, s_bounds)

Or we can instantiate a StochDynProgModel that can be solved only by SDP but we
need to define the constraint function and the final cost function::

    function constraints(t, x, u, xi) # return true when there is no constraints ecept state and control bounds
        return true
    end

    function final_cost_function(x)
        return 0
    end

    spmodel = StochDynProgModel(N_STAGES, s_bounds, u_bounds, X0, cost_t,
                                final_cost_function, dynamic, constraints,
                                xi_laws)


Solver
^^^^^^

It remains to define SDP algorithm parameters::

    stateSteps = [1] # discretization steps of the state space
    controlSteps = [0.1] # discretization steps of the control space
    infoStruct = "HD" # noise at time t is known before taking the decision at time t
    paramSDP = SDPparameters(spmodel, stateSteps, controlSteps, infoStruct)


Now, we solve the problem by computing Bellman values::

    Vs = solve_DP(spmodel,paramSDP, 1)

:code:`V` is an array storing the value functions

We have an exact lower bound given by :code:`V` with the function::

    value_sdp = StochDynamicProgramming.get_bellman_value(spmodel,paramSDP,Vs)


Find optimal controls
=====================

Once Bellman functions are computed, we can control our system over assessments scenarios, without assuming knowledge of the future.

We build 1000 scenarios according to the laws stored in :code:`xi_laws`::

    scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws,1000)

We compute 1000 simulations of the system over these scenarios::

    costsdp, states, controls =sdp_forward_simulation(spmodel,paramSDP,scenarios,Vs)

:code:`costsdp` returns the costs for each scenario, :code:`states` the simulation of each state variable along time, for each scenario, and
:code:`controls` returns the optimal controls for each scenario

