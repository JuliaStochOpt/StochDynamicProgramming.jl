#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Test SDDP with LQR example (quadratic cost)
#############################################################################

include("../src/simulate.jl")
include("../src/SDDP.jl")
include("../src/SDDPoptimize.jl")

using CPLEX
using JuMP

N_STAGES = 20
N_SCENARIOS = 1
A = [1 1; 0 1]
B = [0, 1]

RHO = .3
SIGMA = .01


function cost_t(x, u, w)
    return x[1]^2 + RHO*u[1]^2
end


function dynamic(x, u, w)
    return A*x + B*u + w
end


function init_problem()
    # Instantiate model:
    x0 = 0
    law = Normal(0, SIGMA)
    model = SDDP.LinearDynamicLinearCostSPmodel(N_STAGES, 1, 2, 2, x0, cost_t, dynamic, law)
    solver = CplexSolver(CPX_PARAM_SIMDISPLAY=0)
    params = SDDP.SDDPparameters(solver, N_SCENARIOS)

    return model, params
end


function solve_lqr()
    model, params = init_problem()
    V = optimize(model, params)

    aleas = simulate_scenarios(model.noises ,(1, model.stageNumber, model.dimNoises))

    costs, stocks = forward_simulations(model, params, V, 1, aleas)
    println(stocks)
    println(costs)

end

@time solve_lqr()
