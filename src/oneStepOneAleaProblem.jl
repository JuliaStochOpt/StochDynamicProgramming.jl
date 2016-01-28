#  Copyright 2014, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Model and solve the One-Step One Alea problem in different settings
# - used to compute the optimal control (in forward phase / simulation)
# - used to compute the cuts in the Backward phase
#############################################################################

using JuMP
using CPLEX

include("SDDP.jl")

"""
Solve the Bellman equation at time t starting at state x under alea xi
with the current evaluation of Vt+1

The function solve
min_u current_cost(t,x,u,xi) + current_Bellman_Value_{t+1}(dynamic(t,x,u,xi))
and can return the optimal control and a subgradient of the value of the
problem with respect to the initial state x


Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- V (bellmanFunctions)
    the current estimation of Bellman's functions

- t (int)
    time step at which the problem is solved

- xt (Array{Float})
    current starting state

- xi (Array{float})
    current noise value

- returnOptNextStage (Bool)
    return the optimal state at t+1

- returnOptcontrol (Bool)
    return the optimal control

- returnSubgradient (Bool)
    return the subgradient

- returnCost (Bool)
    return the value of the problem

TODO: update returns

Returns (according to the last parameters):
- costs (Array{float,1})
    an array of the simulated costs
- stocks (Array{float})
    the simulated stock trajectories. stocks(k,t,:) is the stock for scenario k at time t.
- controls (Array{float})
    the simulated controls trajectories. controls(k,t,:) is the control for scenario k at time t.
"""
function solve_one_step_one_alea(model, #::SDDP.LinearDynamicLinearCostSPmodel,
                                 param, #::SDDP.SDDPparameters,
                                 V, #::Vector{SDDP.PolyhedralFunction},
                                 t, #::Int64,
                                 xt, #::Vector{Float64},
                                 xi)

    lambdas = V[t].lambdas
    betas = V[t].betas
    # Get JuMP model stored in SDDPparameters:
    m = Model(solver=CplexSolver(CPX_PARAM_SIMDISPLAY=0))
    @defVar(m, x)
    @defVar(m, u)
    @defVar(m, alpha)

    @addConstraint(m, state_constraint, x .== xt)

    for i=1:V[t].numCuts
        @addConstraint(m, betas[i] + lambdas[i]*(model.dynamics(x, u, xi)-xt) .<= alpha)
    end

    @setObjective(m, Min, model.costFunctions(x, u, xi) + alpha)

    status = solve(m)
    solved = (string(status) == "Optimal")

    if solved
        uopt = getValue(u)
        result = SDDP.NextStep(model.dynamics(xt, uopt, xi),
                          uopt,
                          getDual(state_constraint),
                          getObjectiveValue(m))
    else
        # If no solution is found, then return nothing
        result = nothing
    end

    return solved, result
end
