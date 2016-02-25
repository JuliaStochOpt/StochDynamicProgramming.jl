#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# run unit-tests
#############################################################################

push!(LOAD_PATH, "src")

using StochDynamicProgramming
using Distributions, Clp, FactCheck, JuMP


# Test simulate.jl
facts("Probability functions") do
    support = [1, 2, 3]
    proba = [.2 .5 .3]

    law = NoiseLaw(support, proba)
    @fact typeof(law) --> NoiseLaw
    @fact law.supportSize --> 3

    dims = (2, 2, 1)
    scenarios = simulate_scenarios([law, law], dims)
    @fact typeof(scenarios) --> Array{Float64, 3}
    @fact size(scenarios) --> (2, 2, 1)

    scenarios2 = simulate_scenarios(Normal(), dims)
    @fact typeof(scenarios2) --> Array{Float64, 3}
end


facts("Utils functions") do
    # Test extraction of vector in array:
    arr = rand(4, 4, 2)
    vec = StochDynamicProgramming.extract_vector_from_3Dmatrix(arr, 2, 1)
    @fact typeof(vec) --> Vector{Float64}
    @fact size(vec) --> (2,)

    # Test upper bound calculation:
    cost = rand(10)
    upb = StochDynamicProgramming.upper_bound(cost)
    tol = sqrt(2) * erfinv(2*.975 - 1)
    @fact upb --> mean(cost) + tol*std(cost)/sqrt(length(cost))

    # Test stopping criterion:
    @fact StochDynamicProgramming.test_stopping_criterion(1., .999, 0.01) --> true
end


# Test SDDP with a one dimensional stock:
facts("SDDP algorithm: 1D case") do
    solver = ClpSolver()

    # SDDP's tolerance:
    epsilon = .05
    # maximum number of iterations:
    max_iterations = 2
    # number of scenarios in forward and backward pass:
    n_scenarios = 10
    # number of aleas:
    n_aleas = 5
    # number of stages:
    n_stages = 2

    # define dynamic:
    function dynamic(t, x, u, w)
        return [x[1] - u[1] - u[2] + w[1]]
    end
    # define cost:
    function cost(t, x, u, w)
        return -u[1]
    end

    # Generate probability laws:
    laws = Vector{NoiseLaw}(n_stages)
    proba = 1/n_aleas*ones(n_aleas)
    for t=1:n_stages
        laws[t] = NoiseLaw([0, 1, 3, 4, 6], proba)
    end

    # set initial position:
    x0 = [10.]
    # set bounds on state:
    x_bounds = [(0., 100.)]
    # set bounds on control:
    u_bounds = [(0., 7.), (0., Inf)]

    # Instantiate parameters of SDDP:
    params = StochDynamicProgramming.SDDPparameters(solver, n_scenarios,
                                                    epsilon, max_iterations)


    context("Linear cost") do
        # Instantiate a SDDP linear model:
        model = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(n_stages,
                                                    u_bounds, x0,
                                                    cost,
                                                    dynamic, laws)
        set_state_bounds(model, x_bounds)


        # Compute bellman functions with SDDP:
        V, pbs = solve_SDDP(model, params, false)
        @fact typeof(V) --> Vector{StochDynamicProgramming.PolyhedralFunction}
        @fact typeof(pbs) --> Vector{JuMP.Model}

        # Test if the first subgradient has the same dimension as state:
        @fact length(V[1].lambdas[1, :]) --> model.dimStates
        @fact V[1].numCuts --> n_stages*n_scenarios + 1
        @fact length(V[1].lambdas[:, 1]) --> n_stages*n_scenarios + 1

        # Test upper bounds estimation with Monte-Carlo:
        n_simulations = 100
        upb = StochDynamicProgramming.estimate_upper_bound(model, params, V, pbs,
                                                           n_simulations)[1]
        @fact typeof(upb) --> Float64

        # Test a simulation upon given scenarios:
        aleas = simulate_scenarios(model.noises,
                              (model.stageNumber,
                               params.forwardPassNumber,
                               model.dimNoises))

        sddp_costs, stocks = forward_simulations(model, params, V, pbs, aleas)

        # Compare sddp cost with those given by extensive formulation:
        ef_cost = StochDynamicProgramming.extensive_formulation(model,params)
        @fact typeof(ef_cost) --> Float64

        @fact mean(sddp_costs) --> roughly(ef_cost)
    end


    context("Piecewise linear cost") do
        # Test Piecewise linear costs:
        model = StochDynamicProgramming.PiecewiseLinearCostSPmodel(n_stages,
                                                    u_bounds, x0,
                                                    [cost],
                                                    dynamic, laws)
        set_state_bounds(model, x_bounds)
        V, pbs = solve_SDDP(model, params, false)
    end

end


# Test SDDP with a two-dimensional stock:
facts("SDDP algorithm: 2D case") do
    solver = ClpSolver()

    # SDDP's tolerance:
    epsilon = .05
    # maximum number of iterations:
    max_iterations = 2
    # number of scenarios in forward and backward pass:
    n_scenarios = 10
    # number of aleas:
    n_aleas = 5
    # number of stages:
    n_stages = 2

    # define dynamic:
    function dynamic(t, x, u, w)
        return [x[1] - u[1] - u[2] + w[1], x[2] - u[4] - u[3] + u[1] + u[2]]
    end
    # define cost:
    function cost(t, x, u, w)
        return -u[1] - u[3]
    end

    # Generate probability laws:
    laws = Vector{NoiseLaw}(n_stages)
    proba = 1/n_aleas*ones(n_aleas)
    for t=1:n_stages
        laws[t] = NoiseLaw([0, 1, 3, 4, 6], proba)
    end

    # set initial position:
    x0 = [10., 10]
    # set bounds on state:
    x_bounds = [(0., 100.), (0, 100)]
    # set bounds on control:
    u_bounds = [(0., 7.), (0., Inf), (0., 7.), (0., Inf)]

    # Instantiate parameters of SDDP:
    params = StochDynamicProgramming.SDDPparameters(solver, n_scenarios,
                                                    epsilon, max_iterations)
    context("Linear cost") do
        # Instantiate a SDDP linear model:
        model = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(n_stages,
                                                    u_bounds, x0,
                                                    cost,
                                                    dynamic, laws)
        set_state_bounds(model, x_bounds)


        # Compute bellman functions with SDDP:
        V, pbs = solve_SDDP(model, params, false)
        @fact typeof(V) --> Vector{StochDynamicProgramming.PolyhedralFunction}
        @fact typeof(pbs) --> Vector{JuMP.Model}

        # Test if the first subgradient has the same dimension as state:
        @fact length(V[1].lambdas[1, :]) --> model.dimStates

        # Test upper bounds estimation with Monte-Carlo:
        n_simulations = 100
        upb = StochDynamicProgramming.estimate_upper_bound(model, params, V, pbs,
                                                           n_simulations)[1]
        @fact typeof(upb) --> Float64


         # Test a simulation upon given scenarios:
        aleas = simulate_scenarios(model.noises,
                              (model.stageNumber,
                               params.forwardPassNumber,
                               model.dimNoises))

        sddp_costs, stocks = forward_simulations(model, params, V, pbs, aleas)

        # Compare sddp cost with those given by extensive formulation:
        ef_cost = StochDynamicProgramming.extensive_formulation(model,params)
        @fact typeof(ef_cost) --> Float64

        @fact mean(sddp_costs) --> roughly(ef_cost)
    end
end
