#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define the Forward / Backward iterations of the SDDP algorithm
#############################################################################

include("SDDP.jl")

"""
Make a forward pass of the algorithm

Simulate a scenario of noise and compute an optimal trajectory on this
scenario according to the current value functions.

Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm
- V (bellmanFunctions)
    the current estimation of Bellman's functions

- forwardPassNumber (int)
    number of forward simulation

- xi (Array{float})
    the noise scenarios on which we simulate, each line being one scenario.
    Generated if not given.

- returnCosts (Bool)
    return the cost of each simulated scenario if true

- returnStocks (Bool)
    return the trajectory of the stocks if true

- returnControls (Bool)
    return the trajectory of controls if true


Returns (according to the last parameters):
- costs (Array{float,1})
    an array of the simulated costs
- stocks (Array{float})
    the simulated stock trajectories. stocks(k,t,:) is the stock for scenario k at time t.
- controls (Array{float})
    the simulated controls trajectories. controls(k,t,:) is the control for scenario k at time t.
"""
function forward_simulations(model::SDDP.SPModel,
                            param::SDDP.SDDPparameters,
                            V::Vector{SDDP.PolyhedralFunction},
                            forwardPassNumber::Int64,
                            xi = nothing,
                            returnCosts::Bool = true,
                            returnStocks::Bool= true,
                            returnControls::Bool = false)

    # TODO simplify if returnStocks=false
    # TODO stock Controls

    # TODO declare stock as an array of states
    # specify initial state stocks[k,0]=x0
    # TODO generate scenarios xi
    if returnCosts
        costs = zeros(k)
    end

    for k = 1:forwardPassNumber #TODO can be parallelized + some can be dropped if too long

        for t=0:T-1 #TODO get T
            stocks[k,t+1], opt_control = solveOneStepOneAlea(t,stocks[k,t],xi[k,t],
                                        returnOptNextStage=true,
                                        returnOptControl=true,
                                        returnSubgradient=false,
                                        returnCost=false);
            if returnCosts
                costs[k] += costFunction(t,stocks[k,t],opt_control,xi[k,t]) #TODO
            end
        end
    end
    return costs,stocks # adjust according to what is asked
end



"""
Add to Vt a cut of the form Vt >= beta + <lambda,.>

Parameters:
- Vt (bellmanFunction)
    Current lower approximation of the Bellman function at time t
- beta (Float)
    affine part of the cut to add
- lambda (Array{float,1})
    subgradient of the cut to add
"""
function addcut!(Vt::SDDP.PolyhedralFunction, beta::Float64, lambda)
    #TODO add >= beta + <lambda,.>,
    Vt.lambdas = vcat(Vt.lambdas, lambda)
    Vt.betas = vcat(Vt.betas, beta)
    Vt.numCuts += 1
end



"""
Make a backward pass of the algorithm

For t:T-1 -> 0, compute a valid cut of the Bellman function
Vt at the state given by stockTrajectories and add them to
the current estimation of Vt.

Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- V (bellmanFunctions)
    the current estimation of Bellman's functions

- stockTrajectories (Array{Float64,3})
    stockTrajectories[k,t,:] is the vector of stock where the cut is computed
    for scenario k and time t.

Return nothing
"""
function backward_pass(model::SDDP.SPModel,
                      param::SDDP.SDDPparameters,
                      V::Array{SDDP.PolyhedralFunction},
                      stockTrajectories)

    for t = T-1:0
        for k = 1:20 #TODO number of trajectories
            cost = zeros(1);
            subgradient = zeros(dimStates[t]);#TODO access
            for w in 1:nXi[t] #TODO number of alea at t + can be parallelized
                subgradientw, costw = solveOneStepOneAlea(t,
                                            stockTrajectories[k,t],
                                            xi[t,],
                                            returnOptNextStage=false,
                                            returnOptControl=false,
                                            returnSubgradient=true,
                                            returnCost=true)
                cost += prob[w, t]*costw #TODO obtain probability
                subgradientw += prob[w, t]*subgradientw #TODO
            end
            beta = cost - subgradientw*stockTrajectories[k, t, :] #TODO dot product not working
            addCut!(V[t], beta, subgradientw) #TODO access of V[t]
        end
    end
end
