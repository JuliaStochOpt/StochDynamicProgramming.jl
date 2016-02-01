#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define the Forward / Backward iterations of the SDDP algorithm
#############################################################################

using SDDP
include("oneStepOneAleaProblem.jl")
include("utility.jl")

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
function forward_simulations(model, #::SDDP.LinearDynamicLinearCostSPmodel,
                            param, #::SDDP.SDDPparameters,
                            V, #::Vector{SDDP.PolyhedralFunction},
                            forwardPassNumber::Int64,
                            xi::Array{Float64, 3},
                            returnCosts=true,
                            display=false)

    # TODO: verify that loops are in the same order
    # TODO: add a trick to return cost
    # TODO simplify if returnStocks=false
    # TODO stock Controls
    T = model.stageNumber
    stocks = zeros(param.forwardPassNumber, T, model.dimStates)
    # TODO declare stock as an array of states
    # specify initial state stocks[k,0]=x0
    # TODO generate scenarios xi
    costs = nothing
    if returnCosts
        costs = zeros(param.forwardPassNumber)
    end

    #TODO: can be parallelized + some can be dropped if too long
    for k = 1:param.forwardPassNumber

        for t=1:T-1
            state_t = extract_vector_from_3Dmatrix(stocks, t, k)
            alea_t = extract_vector_from_3Dmatrix(xi, t, k)

            status, nextstep = solve_one_step_one_alea(
                                            model,
                                            param,
                                            V,
                                            t,
                                            state_t,
                                            alea_t)

            stocks[k, t+1, :] = nextstep.next_state
            opt_control = nextstep.optimal_control
            if display
                println(sizeof(opt_control))
            end
            #TODO: implement returnCosts
            if returnCosts
                costs[k] += model.costFunctions(state_t, opt_control, alea_t)

                # costs[k] += nextstep.cost
            end
        end
    end
    return costs, stocks
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
function add_cut!(Vt, beta::Float64, lambda::Array{Float64,1})
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
function backward_pass(model, #::SDDP.SPModel,
                      param, #::SDDP.SDDPparameters,
                      V, #::Array{SDDP.PolyhedralFunction, 1},
                      stockTrajectories,
                      law, #::NoiseLaw,
                      init=false)

    T = model.stageNumber

    isNoiseLaw = (typeof(law) == NoiseLaw)
    if isNoiseLaw
        nXi = law.supportSize
    else
        nXi = size(law)[1]
    end
    subgradient = 0
    state_t = zeros(Float64, model.dimStates)

    for t = T-1:-1:1
        for k = 1:param.forwardPassNumber
            cost = zeros(1);
            subgradient = zeros(model.dimStates);#TODO access

            for w in 1:nXi #TODO: number of alea at t + can be parallelized
                state_t = extract_vector_from_3Dmatrix(stockTrajectories, t, k)

                if isNoiseLaw
                    alea_t  = [law.support[w]]
                    proba_t = law.proba[w]
                else
                    alea_t = extract_vector_from_3Dmatrix(law, t, k)
                    proba_t = 1/nXi
                end

                nextstep = solve_one_step_one_alea(model,
                                                   param,
                                                   V,
                                                   t,
                                                   state_t,
                                                   alea_t)[2]
                subgradientw = nextstep.sub_gradient
                costw = nextstep.cost

                #TODO: obtain probability cost += prob[w, t] * costw
                #TODO: add non uniform distribution laws
                #TODO: compute probability of costs outside this loop
                cost += proba_t * costw
                subgradient += proba_t * subgradientw
            end

            beta = cost - dot(subgradient, state_t)

            if init
                println(subgradient)
                V[t] = SDDP.PolyhedralFunction(beta, reshape(subgradient, model.dimStates, 1), 1)
            else
                add_cut!(V[t], beta[1], subgradient)
            end

        end
    end
end
