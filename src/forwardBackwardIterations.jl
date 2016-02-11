#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define the Forward / Backward iterations of the SDDP algorithm
#############################################################################

using JuMP
include("oneStepOneAleaProblem.jl")
include("utility.jl")
include("objects.jl")

"""
Make a forward pass of the algorithm

Simulate a scenario of noise and compute an optimal trajectory on this
scenario according to the current value functions.

Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- V (PolyhedralFunction)
    the current estimation of Bellman's functions

- solverProblems (Array{JuMP.Model})
    Linear model used to approximate each value function

- forwardPassNumber (int)
    number of forward simulation

- xi (Array{float})
    the noise scenarios on which we simulate, each line being one scenario.
    Generated if not given.

- returnCosts (Bool)
    return the cost of each simulated scenario if true

- init (Bool)
    Specify if the problem must be initialized
    (ie cuts are empty)

- display (Bool)
    If specified, display results in shell


Returns (according to the last parameters):
- costs (Array{float,1})
    an array of the simulated costs

- stocks (Array{float})
    the simulated stock trajectories. stocks(k,t,:) is the stock for
    scenario k at time t.


"""
function forward_simulations(model::SPModel,
                            param::SDDPparameters,
                            V::Vector{PolyhedralFunction},
                            solverProblems::Vector{JuMP.Model},
                            forwardPassNumber::Int64,
                            xi::Array{Float64, 3},
                            returnCosts=true::Bool,
                            init=false::Bool,
                            display=false::Bool)

    # TODO: verify that loops are in the same order
    T = model.stageNumber
    stocks = zeros(param.forwardPassNumber, T, model.dimStates)
    controls = zeros(param.forwardPassNumber, T, model.dimControls)
    for i in 1:forwardPassNumber
        stocks[i, 1, :] = model.initialState
    end

    costs = nothing
    if returnCosts
        costs = zeros(param.forwardPassNumber)
    end

    for k = 1:param.forwardPassNumber

        for t=1:T-1
            state_t = extract_vector_from_3Dmatrix(stocks, t, k)
            alea_t = extract_vector_from_3Dmatrix(xi, k, t)

            status, nextstep = solve_one_step_one_alea(
                                            model,
                                            param,
                                            solverProblems[t],
                                            t,
                                            state_t,
                                            alea_t,
                                            init)

            stocks[k, t+1, :] = nextstep.next_state
            opt_control = nextstep.optimal_control
            controls[k, t, :] = opt_control
            if display
                println(opt_control)
            end

            if returnCosts
                costs[k] += nextstep.cost - nextstep.cost_to_go
            end
        end
    end
    return costs, stocks, controls
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
function add_cut!(model::SPModel, problem::JuMP.Model,
                  t::Int64, Vt::PolyhedralFunction,
                  beta::Float64, lambda::Array{Float64,1})
    Vt.lambdas = vcat(Vt.lambdas, lambda)
    Vt.betas = vcat(Vt.betas, beta)
    Vt.numCuts += 1

    alpha = getVar(problem, :alpha)
    x = getVar(problem, :x)
    u = getVar(problem, :u)
    w = getVar(problem, :w)

    @addConstraint(problem, beta + dot(lambda, model.dynamics(t, x, u, w)) <= alpha)
end

"""
Update linear problem with cuts stored in given PolyhedralFunction.

Parameters:
- model (SPModel)
    Store the problem definition

- problem (JuMP.Model)
    Linear problem used to approximate the value functions

- Vt (PolyhedralFunction)
    Store values of each cut

"""
function add_constraints_with_cut!(model::SPModel, problem::JuMP.Model,
                                   t::Int64, Vt::PolyhedralFunction)
    for i in 1:Vt.numCuts

        alpha = getVar(problem, :alpha)
        x = getVar(problem, :x)
        u = getVar(problem, :u)
        w = getVar(problem, :w)
        @addConstraint(problem, Vt.betas[i] + Vt.lambdas[i]*model.dynamics(t, x, u, w) .<= alpha)
    end
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

- V (Array{PolyhedralFunction})
    the current estimation of Bellman's functions

- solverProblems (Array{JuMP.Model})
    Linear model used to approximate each value function

- stockTrajectories (Array{Float64,3})
    stockTrajectories[k,t,:] is the vector of stock where the cut is computed
    for scenario k and time t.

- law (Array{NoiseLaw})
    Conditionnal distributions of perturbation, for each timestep

Return nothing

"""
function backward_pass(model::SPModel,
                      param::SDDPparameters,
                      V::Array{PolyhedralFunction, 1},
                      solverProblems::Vector{JuMP.Model},
                      stockTrajectories::Array{Float64, 3},
                      law, #::NoiseLaw,
                      init=false)

    T = model.stageNumber

    subgradient = 0
    state_t = zeros(Float64, model.dimStates)

    for t = T-1:-1:2
        for k = 1:param.forwardPassNumber
            cost = zeros(1);
            subgradient = zeros(model.dimStates)

            for w in 1:law[t].supportSize
                state_t = extract_vector_from_3Dmatrix(stockTrajectories, t, k)

                alea_t  = collect(law[t].support[:, w])
                proba_t = law[t].proba[w]

                nextstep = solve_one_step_one_alea(model,
                                                   param,
                                                   solverProblems[t],
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
                V[t] = SDDP.PolyhedralFunction(beta,
                                               reshape(subgradient,
                                                       model.dimStates,
                                                       1), 1)
                add_constraints_with_cut!(model, solverProblems[t-1], t, V[t])
            else
                subgradient = Array{Float64}(subgradient)
                add_cut!(model, solverProblems[t-1], t, V[t], beta[1], subgradient)
            end

        end
    end
end
