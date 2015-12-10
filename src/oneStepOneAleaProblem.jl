#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Model and solve the One-Step One Alea problem in different settings
# - used to compute the optimal control (in forward phase / simulation)
# - used to compute the cuts in the Backward phase
#############################################################################




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

- x (Array{Float})
    current starting state 
    
- xi (Array{float}) 
    current noise value

- returnOptNextStep (Bool)
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
function solveOneStepOneAlea(model::LinearDynamicLinearCostSPmodel,
                            param::SDDPparameters,
                            V::Vector{PolyhedralFunction},
                            t,
                            x::Vector{AbstractFloat},
                            xi::Vector{AbstractFloat},
                            returnOptNextStep::Bool=false, 
                            returnOptControl::Bool=false,
                            returnSubgradient::Bool=false,
                            returnCost::Bool=false)
    
    #TODO call the right following function
    # return (optNextStep, optControl, subgradient, cost) #depending on which is asked
end


