srand(2713)
push!(LOAD_PATH, "../src")
# include("../src/objects.jl")
# include("../src/SDPoptimize.jl")

using StochDynamicProgramming, JuMP, Clp, Distributions

const SOLVER = ClpSolver()
# const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0)

const EPSILON = .05
const MAX_ITER = 20

# Define number of stages and scenarios:
const N_STAGES = 3
const N_SCENARIOS = 3

# Define time horizon:
const TF = N_STAGES-1

# Randomnly generate a cost scenario fixed for the whole problem:
const COST = -66*2.7*(1 + .5*(rand(TF) - .5))

# Define bounds for states and controls:
const VOLUME_MAX = 100
const VOLUME_MIN = 0

const CONTROL_MAX = 80
const CONTROL_MIN = 0

# Define realistic bounds for aleas:
const W_MAX = 70
const W_MIN = 0

# Randomly generate two deterministic scenarios for rain
alea_year1 =round(Int, (W_MAX-W_MIN)*rand(TF)-W_MIN)

# Define initial states of both dams:
const X0 = [50]


# Define dynamic of the dams:
function dynamic(t, x, u, w)
    return [x[1] - u[1] - u[2] + w[1]]
end

# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COST[t] * (u[1])
end

function finalCostFuncion(x)
    return 0
end

function constraints(t, x, x1, u, w)

    Bu = (x1[1]<=VOLUME_MAX)&(x[1]<=VOLUME_MAX)
    Bl = (x1[1]>=VOLUME_MIN)&(x[1]>=VOLUME_MIN)

    return Bu&Bl

end


"""Solve the problem with a solver, supposing the aleas are known
in advance."""
function solve_determinist_problem()
    m = Model(solver=SOLVER)


    @defVar(m,  VOLUME_MIN  <= x1[1:(TF+1)]  <= VOLUME_MAX)
    @defVar(m,  CONTROL_MIN <= u1[1:TF]  <= CONTROL_MAX)

    @setObjective(m, Min, sum{COST[i]*(u1[i]), i = 1:TF})

    for i in 1:TF
        @addConstraint(m, x1[i+1] - x1[i] + u1[i] - alea_year1[i] == 0)
    end

    @addConstraint(m, x1[1] == X0[1])

    status = solve(m)
    println(status)
    println(getObjectiveValue(m))
    return getValue(u1), getValue(x1)
end


"""Build admissible scenarios for water inflow over the time horizon."""
function build_scenarios(n_scenarios::Int64)
    scenarios = zeros(n_scenarios, TF)

    for scen in 1:n_scenarios
        scenarios[scen, :] = round(Int, (W_MAX-W_MIN)*rand(TF)+W_MIN)
    end
    return scenarios
end


"""Build probability distribution at each timestep based on N scenarios.

Return a Vector{NoiseLaw}"""
function generate_probability_laws()
    aleas = build_scenarios(N_SCENARIOS)

    laws = Vector{NoiseLaw}(TF)

    # uniform probabilities:
    proba = 1/N_SCENARIOS*ones(N_SCENARIOS)

    for t=1:TF
        laws[t] = NoiseLaw(aleas[:, t], proba)
    end

    return laws
end


"""Instantiate the problem."""
function init_problem()

    x0 = X0
    aleas = generate_probability_laws()

    x_bounds = [(VOLUME_MIN, VOLUME_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (0, VOLUME_MAX)]

    N_CONTROLS = 2
    N_STATES = 1
    N_ALEAS = 1

    model = LinearDynamicLinearCostSPmodel(TF+1,
                                        N_CONTROLS,
                                        N_STATES,
                                        N_ALEAS,
                                        x_bounds,
                                        u_bounds,
                                        x0,
                                        cost_t,
                                        dynamic,
                                        aleas)

    solver = SOLVER

    params = SDDPparameters(solver, N_SCENARIOS, EPSILON, MAX_ITER)

    return model, params
end

function init_problem_sdp()

    x0 = X0
    aleas = generate_probability_laws()

    x_bounds = [(VOLUME_MIN, VOLUME_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (0, VOLUME_MAX)]

    N_CONTROLS = 2
    N_STATES = 1
    N_ALEAS = 1

    stateSteps = [1, 1]
    controlSteps = [1, 1, 1, 1]
    stateVariablesSizes = [(VOLUME_MAX-VOLUME_MIN)+1]
    controlVariablesSizes = [(CONTROL_MAX-CONTROL_MIN)+1, (VOLUME_MAX)+1]
    totalStateSpaceSize = stateVariablesSizes[1]
    totalControlSpaceSize = controlVariablesSizes[1]*controlVariablesSizes[2]
    monteCarloSize = 10

    model = DPSPmodel(TF,
                    N_CONTROLS,
                    N_STATES,
                    N_ALEAS,
                    x_bounds,
                    u_bounds,
                    x0,
                    cost_t,
                    finalCostFuncion,
                    dynamic,
                    constraints,
                    aleas)

    params = SDPparameters(stateSteps, controlSteps, totalStateSpaceSize, totalControlSpaceSize, stateVariablesSizes, controlVariablesSizes, monteCarloSize)

    return model, params
end



"""Solve the problem."""
function solve_dams(display=false)
    model, params = init_problem()

    V, pbs = optimize(model, params, display)

    params.forwardPassNumber = 1

    aleas = simulate_scenarios(model.noises,
                              (model.stageNumber,
                               params.forwardPassNumber,
                               model.dimNoises))

    costs, stocks = forward_simulations(model, params, V, pbs, 1, aleas)

    println("SDDP cost: ", costs)
    return stocks, V
end


"""Solve the problem."""
function solve_dams_sdp(display=false)
    model, params = init_problem_sdp()

    law = model.noises

    V, Pi = sdp_optimize_DH(model, params, display)

    scenar = Array(Array, TF)

    for t in 1:TF
        scenar[t] = sampling(law, t)
    end

    costs, stocks = sdp_forward_simulation_DH(model, params, scenar, X0, V, Pi, true)

    println("SDP cost: ", costs)
    return stocks, V
end

function solve_dams_sdp_HD(display=false)
    model, params = init_problem_sdp()

    law = model.noises

    V = sdp_optimize_HD(model, params, display)

    scenar = Array(Array, TF)

    for t in 1:TF
        scenar[t] = sampling(law, t)
    end

    costs, stocks = sdp_forward_simulation_HD(model, params, scenar, X0, V, true)

    println("SDP cost: ", costs)
    return stocks, V
end


function compare_SDP_SDDP(display=false)

    modelSDP, paramsSDP = init_problem_sdp()

    lawSDP = modelSDP.noises

    VSDP = sdp_optimize_HD(modelSDP, paramsSDP, display)
    #VSDPDH, Pi = sdp_optimize_DH(modelSDP, paramsSDP, display)

    model, params = init_problem()

    V, pbs = optimize(model, params, display)

    params.forwardPassNumber = 1

    aleas = simulate_scenarios(model.noises,
                              (model.stageNumber,
                               params.forwardPassNumber,
                               model.dimNoises))

    costsSDP, stocksSDP, uSDP = sdp_forward_simulation_HD(modelSDP, paramsSDP, aleas, X0, VSDP, display)
    #costsSDPDH, stocksSDPDH, uSDPDH = sdp_forward_simulation_DH(modelSDP, paramsSDP, aleas, X0, VSDPDH, Pi, display)

    costs, stocks, u = forward_simulations(model, params, V, pbs, 1, aleas)

    println("SDDP cost: ", costs)
    println("SDP cost HD: ", costsSDP)
    #println("SDP cost DH: ", costsSDPDH)

    return stocks, stocksSDP, u, uSDP, aleas

end