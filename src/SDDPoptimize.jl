#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  the actual optimization function
#
#############################################################################


include("forwardBackwardIterations.jl")
include("utility.jl")
include("simulate.jl")

"""
TODO: add docstring
TODO: move initialize in proper module
TODO: fix initialize

"""
function get_null_value_functions_array(model::SDDP.SPModel)

    V = Vector{SDDP.PolyhedralFunction}(model.stageNumber)
    for t = 1:model.stageNumber
        V[t] = get_null_value_functions()
    end

    return V
end

function build_models(model::SDDP.SPModel, param::SDDP.SDDPparameters)

    models = Vector{JuMP.Model}(model.stageNumber)


    for t = 1:model.stageNumber
      m = Model(solver=param.solver)

      @defVar(m, 0<= x[1:1] <= 100)
      @defVar(m, 0 <= u[1:2] <= 7)
      @defVar(m, alpha)
      @defVar(m, w[1:1])

      # @addConstraints(m, 0 .<= model.dynamics(x, u, w))
      # @addConstraint(m, -100 .<= -model.dynamics(x, u, w))

      @setObjective(m, Min, model.costFunctions(t, x, u, w) + alpha)

      models[t] = m

    end
    return models
end


function initialize_value_functions( model::SDDP.LinearDynamicLinearCostSPmodel,
                                     param::SDDP.SDDPparameters,
                        )

    solverProblems = build_models(model, param)
    V_null = get_null_value_functions_array(model)
    V = Array{SDDP.PolyhedralFunction}(model.stageNumber)

    aleas = simulate_scenarios(model.noises,
                               (model.stageNumber,
                                param.forwardPassNumber,
                                model.dimNoises))

    n = param.forwardPassNumber

    V[end] = SDDP.PolyhedralFunction(zeros(1), zeros(1, 1), 1)

    stockTrajectories = forward_simulations(model,
                        param,
                        V_null,
                        solverProblems,
                        n,
                        aleas)[2]

    backward_pass(model,
                  param,
                  V,
                  solverProblems,
                  stockTrajectories,
                  model.noises,
                  true)
    return V
end



"""
Make a forward pass of the algorithm

Simulate a scenario of noise and compute an optimal trajectory on this
scenario according to the current value functions.

Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm


Returns :
- V::Array{PolyhedralFunction}
    the collection of approximation of the bellman functions

"""
function optimize(model::SDDP.SPModel,
                  param::SDDP.SDDPparameters,
                  n_iterations=20)

    # Initialize value functions:
    V = initialize_value_functions(model, param)
    println("Initialize cuts")
    aleas = simulate_scenarios(model.noises ,(model.stageNumber, param.forwardPassNumber , model.dimNoises))
    stopping_test::Bool = false
    iteration_count::Int64 = 0


    n = param.forwardPassNumber

    for i = 1:n_iterations
        stockTrajectories = forward_simulations(model,
                            param,
                            V,
                            n,
                            aleas)[2]

        backward_pass(model,
                      param,
                      V,
                      stockTrajectories,
                      model.noises)
        # TODO: stopping test

        iteration_count+=1;
        println("Pass number ", i)
    end

    return V
end
