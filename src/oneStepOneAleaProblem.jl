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
using SDDP

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
TODO: add types in function parameters

"""
function solve_one_step_one_alea(model, #::SDDP.LinearDynamicLinearCostSPmodel,
                                 param, #::SDDP.SDDPparameters,
                                 V, #::Vector{SDDP.PolyhedralFunction},
                                 t, #::Int64,
                                 xt, #::Vector{Float64},
                                 xi) #::Vector{Float64},

    lambdas = V[t+1].lambdas
    betas = V[t+1].betas
    # println(lambdas, " ", betas)
    # TODO: factorize the definition of the model in PolyhedralFunction
    m = Model(solver=param.solver)
    @defVar(m, 0<= x[1:1] <= 100)
    @defVar(m, 0 <= u[1:2] <= 7)
    @defVar(m, alpha)


    @addConstraint(m, state_constraint, x .== xt)
    @addConstraint(m, 0 <= model.dynamics(x, u, xi) )
    @addConstraint(m, -100 <= -model.dynamics(x, u, xi) )
    # TODO: implement cost function in PolyhedralFunction

    for i=1:V[t+1].numCuts
          @addConstraint(m, betas[i] + lambdas[i]*(x[1] - u[1] - u[2] + xi[1]) <= alpha)
    end

    @setObjective(m, Min, COST[t]*u[1] + alpha)

    status = solve(m)
    solved = (string(status) == "Optimal")

    if solved
        optimalControl = getValue(u)
        # Return object storing results:
        result = SDDP.NextStep(
                          [model.dynamics(xt, optimalControl, xi)],
                          optimalControl,
                          getDual(state_constraint),
                          getObjectiveValue(m))
    else
        # If no solution is found, then return nothing
        result = nothing
    end

    return solved, result
end
