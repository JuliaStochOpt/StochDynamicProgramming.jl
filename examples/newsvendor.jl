#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Test SDDP with the newsvendor case study
#############################################################################

include("../src/SDDPoptimize.jl")
include("../src/simulate.jl")

using Clp
using JuMP

N_STAGES = 20
N_SCENARIOS = 1


function cost_t(x, u, w)
    h = .5
    p = 3

    cost = 0
    if x[1] >= 0
        cost += h*x[1]
    else
        cost += -p*x[1]
    end
    return cost + u[1]
end


function dynamic(x, u, w)
    return x + u -w
end


function init_problem()
    # Instantiate model:
    x0 = 0
    law = NoiseLaw([0., 1., 2., 3.], [.2, .4, .3, .1])
    model = SDDP.LinearDynamicLinearCostSPmodel(N_STAGES, 1, 1, 1, x0, cost_t, dynamic, law)
    solver = ClpSolver()
    params = SDDP.SDDPparameters(solver, N_SCENARIOS)

    return model, params
end


function solve_newsvendor()
    model, params = init_problem()
    V = optimize(model, params)
    law = NoiseLaw([0., 1., 2., 3.], [.2, .4, .3, .1])


    aleas = simulate_scenarios(law ,(1, model.stageNumber, 1))
    costs, stocks = forward_simulations(model, params, V, 1, aleas)
    println(stocks)
    println(costs)

end

@time solve_newsvendor()
