#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
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
include("utility.jl")

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


Returns (according to the last parameters):
- costs (Array{float,1})
    an array of the simulated costs
- stocks (Array{float})
    the simulated stock trajectories. stocks(k,t,:) is the stock for scenario k at time t.
- controls (Array{float})
    the simulated controls trajectories. controls(k,t,:) is the control for scenario k at time t.
"""
function solve_one_step_one_alea(model::LinearDynamicLinearCostSPmodel,
                            param::SDDPparameters,
                            V::Vector{PolyhedralFunction},
                            t,
                            xt::Vector{Float64},
                            xi::Vector{Float64},
                            returnOptNextStage::Bool=false,
                            returnOptControl::Bool=false,
                            returnSubgradient::Bool=false,
                            returnCost::Bool=false)

    # cost = model.costFunctions[t]
    # dynamic = model.dynamics[t]

    lambdas = V.lambdas
    betas = V.betas

    # Get JuMP model stored in SDDPparameters:
    m = SDDPparameters.solver
    @defVar(m, x)
    @defVar(m, u)
    @defVar(m, alpha)

    @addConstraints(m, state_constraint, x = xt)

    cuts_number = V.numCuts

    for i=1:cuts_number
        @addConstraints(m, betas[i] + lambdas[i]*(dynamic(x, u, xi)-xt) <= alpha)
    end

    @setObjective(m, Min, cost_function(x, u, xi) + alpha)

    solve(m)


    result = []
    if (returnOptNextStep)
        uopt = getValue(u)
        result = [result; dynamic(x, u)]
    end
    if (returnOptControl)
        result = [result; getValue(u)]
    end
    if (returnSubgradient)
        lambda = getDual(state_constraint)
        result = [result; lambda]
    end
    if (returnCost)
        beta_opt = getObjectiveValue(m)
        result = [result; beta_opt]
    end

    return result
end
