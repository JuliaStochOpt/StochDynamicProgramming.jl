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


facts("Utility functions") do
    V = StochDynamicProgramming.get_null_value_functions()
    @fact typeof(V) --> PolyhedralFunction
    @fact V.betas[1] --> 0

    arr = rand(4, 4, 2)
    vec = StochDynamicProgramming.extract_vector_from_3Dmatrix(arr, 2, 1)
    @fact typeof(vec) --> Vector{Float64}
    @fact size(vec) --> (2,)
end


facts("SDDP algorithm") do
    solver = ClpSolver()

    # SDDP's tolerance:
    epsilon = .05
    # maximum number of iterations:
    max_iterations = 4
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

    # Instantiate a SDDP linear model:
    model = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(n_stages,
                                                2, 1, 1,
                                                x_bounds, u_bounds, x0,
                                                cost,
                                                dynamic, laws)

    # Instantiate parameters of SDDP:
    params = StochDynamicProgramming.SDDPparameters(solver, n_scenarios,
                                                    epsilon, max_iterations)

    # Compute bellman functions with SDDP:
    V, pbs = optimize(model, params, false)
    @fact typeof(V) --> Vector{StochDynamicProgramming.PolyhedralFunction}
    @fact typeof(pbs) --> Vector{JuMP.Model}

    # Test upper bounds estimation with Monte-Carlo:
    n_simulations = 100
    upb = StochDynamicProgramming.estimate_upper_bound(model, params, V, pbs,
                                                       n_simulations)
    @fact typeof(upb) --> Float64
end
