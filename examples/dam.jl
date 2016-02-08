#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Test SDDP with dam example
# Source: Adrien Cassegrain
#############################################################################


include("../src/SDDP.jl")
include("../src/SDDPoptimize.jl")
include("../src/simulate.jl")

using JuMP
using Clp

SOLVER = ClpSolver()

N_STAGES = 52
N_SCENARIOS = 2

alea_year = Array([7.0 7.0 8.0 3.0 1.0 1.0 3.0 4.0 3.0 2.0 6.0 5.0 2.0 6.0 4.0 7.0 3.0 4.0 1.0 1.0 6.0 2.0 2.0 8.0 3.0 7.0 3.0 1.0 4.0 2.0 4.0 1.0 3.0 2.0 8.0 1.0 5.0 5.0 2.0 1.0 6.0 7.0 5.0 1.0 7.0 7.0 7.0 4.0 3.0 2.0 8.0 7.0])


# FINAL TIME:
TF = 52

# COST:
COST = -66*2.7*(1 + .5*(rand(TF) - .5))

# Constants:
VOLUME_MAX = 100
VOLUME_MIN = 0

CONTROL_MAX = round(Int, .4/7. * VOLUME_MAX) + 1
CONTROL_MIN = 0

W_MAX = round(Int, .5/7. * VOLUME_MAX)
W_MIN = 0
DW = 1

T0 = 1
HORIZON = 52


# Define aleas' space:
N_ALEAS = Int(round(Int, (W_MAX - W_MIN) / DW + 1))
ALEAS = linspace(W_MIN, W_MAX, N_ALEAS)


# Define dynamic of the dam:
function dynamic(x, u, w)
    return x[1] - u[1] - u[2] + w[1]
end

# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COST[t] * u[1]
end


"""Solve the problem with a solver, supposing the aleas are known
in advance."""
function solve_determinist_problem()
    println(alea_year)
    m = Model(solver=SOLVER)

    @defVar(m,  0           <= x[1:(TF+1)]  <= 100)
    @defVar(m,  0.          <= u[1:TF]  <= 7)
    @defVar(m,  0.          <= s[1:TF]  <= 7)

    @setObjective(m, Min, sum{COST[i]*u[i], i = 1:TF})

    for i in 1:TF
        @addConstraint(m, x[i+1] - x[i] + u[i] + s[i] - alea_year[i] == 0)
    end

    @addConstraint(m, x[1] ==0)

    status = solve(m)
    println(status)
    println(getObjectiveValue(m))
    return getValue(u), getValue(x)
end


"""Build aleas probabilities for each month."""
function build_aleas()
    aleas = zeros(N_ALEAS, TF)

    # take into account seasonality effects:
    unorm_prob = linspace(1, N_ALEAS, N_ALEAS)
    proba1 = unorm_prob / sum(unorm_prob)
    proba2 = proba1[N_ALEAS:-1:1]

    for t in 1:TF
        aleas[:, t] = (1 - sin(pi*t/TF)) * proba1 + sin(pi*t/TF) * proba2
    end
    return aleas
end


"""Build an admissible scenario for water inflow."""
function build_scenarios(n_scenarios::Int64, probabilities)
    scenarios = zeros(n_scenarios, TF)

    for scen in 1:n_scenarios
        for t in 1:TF
            Pcum = cumsum(probabilities[:, t])

            n_random = rand()
            prob = findfirst(x -> x > n_random, Pcum)
            scenarios[scen, t] = prob
        end
    end
    return scenarios
end


"""Build probability distribution at each timestep.

Return a Vector{NoiseLaw}"""
function generate_probability_laws()
    aleas = build_scenarios(N_SCENARIOS, build_aleas())

    laws = Vector{NoiseLaw}(N_STAGES)

    # uniform probabilities:
    proba = 1/N_SCENARIOS*ones(N_SCENARIOS)

    for t=1:N_STAGES
        laws[t] = NoiseLaw(aleas[:, t], proba)
    end

    return laws
end


"""Instantiate the problem."""
function init_problem()
    # Instantiate model:
    x0 = 0
    aleas = generate_probability_laws()
    model = SDDP.LinearDynamicLinearCostSPmodel(N_STAGES, 2, 1, 1, (0, 100), (0, 7), x0, cost_t, dynamic, aleas)

    solver = SOLVER
    params = SDDP.SDDPparameters(solver, N_SCENARIOS)

    return model, params
end

"""Solve the problem."""
function solve_dams(display=false)
    model, params = init_problem()

    V, pbs = optimize(model, params, 20, display)
    aleas = simulate_scenarios(model.noises ,(model.stageNumber, params.forwardPassNumber , model.dimNoises))
    params.forwardPassNumber = 1

    costs, stocks = forward_simulations(model, params, V, pbs, 1, aleas)

    println("SDDP cost: ", costs)
    return stocks
end
