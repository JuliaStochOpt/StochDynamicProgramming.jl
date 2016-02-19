srand(2713)
push!(LOAD_PATH, "../src")

using StochDynamicProgramming, JuMP, Clp

const SOLVER = ClpSolver()
# const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0)

const EPSILON = .05
const MAX_ITER = 20

const N_STAGES = 52
const N_SCENARIOS = 10

# FINAL TIME:
const TF = 52

# COST:
const COST = -66*2.7*(1 + .5*(rand(TF) - .5))

# Constants:
const VOLUME_MAX = 100
const VOLUME_MIN = 0

const CONTROL_MAX = round(Int, .4/7. * VOLUME_MAX) + 1
const CONTROL_MIN = 0

const W_MAX = round(Int, .5/7. * VOLUME_MAX)
const W_MIN = 0
const DW = 1

# Randomly generate two deterministic scenarios for rain
alea_year1 =(W_MAX-W_MIN)*rand(TF)-W_MIN
alea_year2 =(W_MAX-W_MIN)*rand(TF)-W_MIN

const T0 = 1
const HORIZON = 52

const X0 = [50, 50]


# Define dynamic of the dam:
function dynamic(t, x, u, w)
    return [x[1] - u[1] - u[3] + w[1], x[2] - u[2] - u[4] + u[1] + u[3] + w[2]]
end

# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COST[t] * (u[1] + u[2])
end


"""Solve the problem with a solver, supposing the aleas are known
in advance."""
function solve_determinist_problem()
    m = Model(solver=SOLVER)


    @defVar(m,  VOLUME_MIN  <= x1[1:(TF+1)]  <= VOLUME_MAX)
    @defVar(m,  VOLUME_MIN  <= x2[1:(TF+1)]  <= VOLUME_MAX)
    @defVar(m,  CONTROL_MIN <= u1[1:TF]  <= CONTROL_MAX)
    @defVar(m,  CONTROL_MIN <= u2[1:TF]  <= CONTROL_MAX)

    @setObjective(m, Min, sum{COST[i]*(u1[i] + u2[i]), i = 1:TF})

    for i in 1:TF
        @addConstraint(m, x1[i+1] - x1[i] + u1[i] - alea_year1[i] == 0)
        @addConstraint(m, x2[i+1] - x2[i] + u2[i] - u1[i] - alea_year1[i] == 0)
    end

    @addConstraint(m, x1[1] == X0[1])
    @addConstraint(m, x2[1] == X0[2])

    status = solve(m)
    println(status)
    println(getObjectiveValue(m))
    return getValue(u1), getValue(x1), getValue(x2)
end


"""Build an admissible scenario for water inflow."""
function build_scenarios(n_scenarios::Int64)
    scenarios = zeros(n_scenarios, TF)

    for scen in 1:n_scenarios
        scenarios[scen, :] = (W_MAX-W_MIN)*rand(TF)+W_MIN
    end
    return scenarios
end


"""Build probability distribution at each timestep.

Return a Vector{NoiseLaw}"""
function generate_probability_laws()
    aleas = zeros(N_SCENARIOS, TF, 2)
    aleas[:, :, 1] = build_scenarios(N_SCENARIOS)
    aleas[:, :, 2] = build_scenarios(N_SCENARIOS)

    laws = Vector{NoiseLaw}(N_STAGES)

    # uniform probabilities:
    proba = 1/N_SCENARIOS*ones(N_SCENARIOS)

    for t=1:N_STAGES
        aleas_t = reshape(aleas[:, t, :], N_SCENARIOS, 2)'
        laws[t] = NoiseLaw(aleas_t, proba)
    end

    return laws
end


"""Instantiate the problem."""
function init_problem()

    x0 = X0
    aleas = generate_probability_laws()

    x_bounds = [(VOLUME_MIN, VOLUME_MAX), (VOLUME_MIN, VOLUME_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (CONTROL_MIN, CONTROL_MAX), (0, Inf), (0, Inf)]

    N_CONTROLS=4
    N_STATES=2
    N_ALEAS=2

    model = LinearDynamicLinearCostSPmodel(N_STAGES,
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


"""Solve the problem."""
function solve_dams(display=false)
    model, params = init_problem()

    V, pbs = optimize(model, params, display)

    aleas = simulate_scenarios(model.noises,
                              (model.stageNumber,
                               params.forwardPassNumber,
                               model.dimNoises))

    params.forwardPassNumber = 1

    costs, stocks = forward_simulations(model, params, V, pbs, 1, aleas)

    println("SDDP cost: ", costs)
    return stocks, V
end
