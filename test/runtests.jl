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

    V = nothing
    model = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(n_stages,
                                                u_bounds, x0,
                                                cost,
                                                dynamic, laws)
    # Generate scenarios for forward simulations:
    aleas = simulate_scenarios(model.noises,
                          (model.stageNumber,
                           params.forwardPassNumber,
                           model.dimNoises))

    sddp_costs = 0
    context("Linear cost") do
        # Instantiate a SDDP linear model:
        set_state_bounds(model, x_bounds)


        # Compute bellman functions with SDDP:
        V, pbs = solve_SDDP(model, params, 0)
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

        sddp_costs, stocks = forward_simulations(model, params, V, pbs, aleas)

        # Compare sddp cost with those given by extensive formulation:
        ef_cost = StochDynamicProgramming.extensive_formulation(model,params)
        @fact typeof(ef_cost) --> Float64

        @fact mean(sddp_costs) --> roughly(ef_cost)
    end

    context("Hotstart") do
        # Test hot start with previously computed value functions:
        V, pbs = solve_SDDP(model, params, 0, V)
        # Test if costs are roughly the same:
        sddp_costs2, stocks = forward_simulations(model, params, V, pbs, aleas)
        @fact mean(sddp_costs) --> roughly(mean(sddp_costs2))
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

    context("Dump") do
        # Dump V in text file:
        StochDynamicProgramming.dump_polyhedral_functions("dump.dat", V)
        # Get stored values:
        Vdump = StochDynamicProgramming.read_polyhedral_functions("dump.dat")

        @fact V[1].numCuts --> Vdump[1].numCuts
        @fact V[1].betas --> Vdump[1].betas
        @fact V[1].lambdas --> Vdump[1].lambdas
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
    V = nothing
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
         params.forwardPassNumber = n_simulations
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


    context("Dump") do
        # Dump V in text file:
        StochDynamicProgramming.dump_polyhedral_functions("dump.dat", V)
        # Get stored values:
        Vdump = StochDynamicProgramming.read_polyhedral_functions("dump.dat")

        @fact V[1].numCuts --> Vdump[1].numCuts
        @fact V[1].betas --> Vdump[1].betas
        @fact V[1].lambdas --> Vdump[1].lambdas
    end
end


facts("Indexation and interpolation for SDP") do
    var = [0.4, 3.7, 1.9]
    low = [0.1, 1.2, 0.5]
    size = [100, 800, 1000]
    steps = [0.1, 0.05, 0.01]
    totalsize = size[1]*size[2]*size[3]

    vart = [0.42, 3.78, 1.932]
    vart2 = [10.0, 3.78, 1.932]

    ind = StochDynamicProgramming.index_from_variable(var, low, size, steps)
    varind = StochDynamicProgramming.variable_from_index(ind , low, size, steps)

    @fact (ind <= totalsize) --> true
    @fact varind --> roughly(var)

    indnn = StochDynamicProgramming.nearest_neighbor(vart , low, size, steps)
    nn = StochDynamicProgramming.variable_from_index(indnn , low, size, steps)
    indnn3 = StochDynamicProgramming.index_from_variable(nn , low, size, steps)
    nn3 = StochDynamicProgramming.variable_from_index(indnn3 , low, size, steps)

    @fact (nn == [0.4, 3.8, 1.93]) --> true

    @fact (indnn == indnn3) --> true

    ind2 = StochDynamicProgramming.index_from_variable(vart2, low, size, steps)
    varind2 = StochDynamicProgramming.variable_from_index(ind2 , low, size, steps)

    @fact (varind2 == [10.0,3.75,1.93]) --> true

    indnn2 = StochDynamicProgramming.nearest_neighbor(vart2 , low, size, steps)
    nn2 = StochDynamicProgramming.variable_from_index(indnn2 , low, size, steps)

    @fact (nn2 == [10, 3.8, 1.93]) --> true

end


facts("SDP algorithm") do

    # Number of timesteps (as we manage the dams over a year, it is equal to the number of weeks):
    TF = 3

    # Capacity of dams:
    VOLUME_MAX = 30
    VOLUME_MIN = 0

    # Specify the maximum flow of turbines:
    CONTROL_MAX = 20
    CONTROL_MIN = -20

    # Some statistics about aleas (water inflow):
    W_MAX = 15
    W_MIN = 0
    DW = 1

    T0 = 1

    # Define aleas' space:
    N_ALEAS = Int(round(Int, (W_MAX - W_MIN) / DW + 1))
    ALEAS = linspace(W_MIN, W_MAX, N_ALEAS);

    N_CONTROLS = 2;
    N_STATES = 2;
    N_NOISES = 1;

    infoStruct = "HD"

    COST = 66*2.7*(1 + .5*(rand(TF) - .5));

    # Define dynamic of the dam:
    function dynamic(t, x, u, w)
        return [x[1] + u[1] + w[1] - u[2], x[2] - u[1]]
    end

    # Define cost corresponding to each timestep:
    function cost_t(t, x, u, w)
        return COST[t] * (u[1])
    end

    function constraints(t, x, u, w)
        return (x[1]<=VOLUME_MAX)&(x[1]>=VOLUME_MIN)&(x[2]<=VOLUME_MAX)&(x[2]>=VOLUME_MIN)
    end

    function finalCostFunction(x)
        return 0.
    end

    """Build admissible scenarios for water inflow over the time horizon."""
    function build_scenarios(n_scenarios::Int64)
        scenarios = zeros(n_scenarios, TF)

        for scen in 1:n_scenarios
            scenarios[scen, :] = (W_MAX-W_MIN)*rand(TF)+W_MIN
        end
        return scenarios
    end

        """Build probability distribution at each timestep based on N scenarios.
    Return a Vector{NoiseLaw}"""
    function generate_probability_laws(N_STAGES, N_SCENARIOS)
        aleas = zeros(N_SCENARIOS, TF, 1)
        aleas[:, :, 1] = build_scenarios(N_SCENARIOS)

        laws = Vector{NoiseLaw}(N_STAGES)

        # uniform probabilities:
        proba = 1/N_SCENARIOS*ones(N_SCENARIOS)

        for t=1:N_STAGES
            aleas_t = reshape(aleas[:, t, :], N_SCENARIOS, 1)'
            laws[t] = NoiseLaw(aleas_t, proba)
        end

        return laws
    end

    N_SCENARIO = 10
    aleas = generate_probability_laws(TF, N_SCENARIO)

    x_bounds = [(VOLUME_MIN, VOLUME_MAX), (VOLUME_MIN, VOLUME_MAX)];
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (VOLUME_MIN, VOLUME_MAX)];

    x0 = [15, 15]

    alea_year = Array([7.0 7.0])

    aleas_scen = zeros(2, 1, 1)
    aleas_scen[:, 1, 1] = alea_year;

    modelSDP = DPSPmodel(TF-1, N_CONTROLS,
                        N_STATES, N_NOISES,
                        x_bounds, u_bounds,
                        x0, cost_t,
                        finalCostFunction, dynamic,
                        constraints, aleas);

    stateSteps = [1,1];
    controlSteps = [1,1];
    monteCarloSize = 10;

    paramsSDP = StochDynamicProgramming.SDPparameters(modelSDP, stateSteps,
                                                     controlSteps,
                                                     monteCarloSize,
                                                     infoStruct);

    context("Solve and simulate using SDP") do

        V_sdp = sdp_optimize(modelSDP, paramsSDP, false);

        @fact size(V_sdp) --> (961,3)

        costs_sdp, stocks_sdp, controls_sdp = sdp_forward_simulation(modelSDP,
                                                                paramsSDP,
                                                                aleas_scen, x0,
                                                                V_sdp, true )

        @fact size(stocks_sdp) --> (3,1,2)
        @fact size(controls_sdp) --> (2,1,2)

    end

end