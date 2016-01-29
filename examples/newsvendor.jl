#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Test SDDP with the newsvendor case study
#############################################################################

include("../src/SDDPoptimize.jl")

using Clp
using JuMP

N_STAGES = 20
N_SCENARIOS = 1


function cost(x, u, w)
    h = 5
    p = .5

    # if x[1] >= 0
    #     return h*x
    # else
    #     return -p*x
    # end
    return h*x
end


function dynamic(x, u, w)
    return x + u -w

end


function init_problem()
    # Instantiate model:
    x0 = 0
    model = SDDP.LinearDynamicLinearCostSPmodel(N_STAGES, 1, 1, x0, cost, dynamic)

    solver = ClpSolver()
    params = SDDP.SDDPparameters(solver, N_SCENARIOS)

    return model, params
end


function solve_newsvendor()
    model, params = init_problem()
    optimize(model, params)
end

@time solve_newsvendor()
