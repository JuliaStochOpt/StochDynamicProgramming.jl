################################################################################
# Test SDDP functions
################################################################################
using StochDynamicProgramming, JuMP, Clp, CutPruners


# Test SDDP with a one dimensional stock:

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
    n_stages = 3

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
    param = StochDynamicProgramming.SDDPparameters(solver,
                                                   passnumber=n_scenarios,
                                                   gap=epsilon,
                                                   max_iterations=max_iterations,
                                                   prune_cuts=0)

    V = nothing
    model = StochDynamicProgramming.LinearSPModel(n_stages, u_bounds,
                                                  x0, cost, dynamic, laws, riskMeasure = WorstCase())


    set_state_bounds(model, x_bounds)
    # Test error if bounds are not well specified:


    # Generate scenarios for forward simulations:
    noise_scenarios = simulate_scenarios(model.noises,param.forwardPassNumber)

    sddp_costs = 0


        # Compute bellman functions with SDDP:
        sddp = solve_SDDP(model, param, 0)


        V = sddp.bellmanfunctions
        # Test if the first subgradient has the same dimension as state:


        # Test upper bounds estimation with Monte-Carlo:
        n_simulations = 100
        upb = StochDynamicProgramming.estimate_upper_bound(model,
                                                           param,
                                                           V,
                                                           sddp.solverinterface,
                                                           n_simulations)[1]


        pbs = sddp.solverinterface
        sddp_costs, stocks = forward_simulations(model, param, pbs,
                                                 noise_scenarios)
        # Test error if scenarios are not given in the right shape:


        # Test computation of optimal control:
        aleas = noise_scenarios[1, 1, :]
        opt = StochDynamicProgramming.get_control(model, param,
                                                  sddp.solverinterface,
                                                  1, model.initialState, aleas)


        # Test display:
        param.compute_ub = 0
        sddp = solve_SDDP(model, param, V, 2)
