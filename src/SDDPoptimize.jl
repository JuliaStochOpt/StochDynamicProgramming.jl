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


function initialize_value_functions( model::SDDP.LinearDynamicLinearCostSPmodel,
                                     param::SDDP.SDDPparameters,
                        )

    V_null = get_null_value_functions_array(model)
    println("ok")
    V = Array{SDDP.PolyhedralFunction}(model.stageNumber)

    aleas = simulate_scenarios([0., 1., 2., 3.],
                               [.2, .4, .3, .1],
                               (param.forwardPassNumber,
                                model.stageNumber, 1))

    n = param.forwardPassNumber

    V[end] = SDDP.PolyhedralFunction(zeros(1), zeros(1, 1), 1)

    stockTrajectories = forward_simulations(model,
                        param,
                        V_null,
                        n,
                        aleas)[2]
    backward_pass(model,
                  param,
                  V,
                  stockTrajectories,
                  aleas,
                  true)
    return V_null
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
                  param::SDDP.SDDPparameters)

    # Initialize value functions:
    V = initialize_value_functions(model, param)
    aleas = simulate_scenarios([0., 1., 2., 3.], [.2, .4, .3, .1],(param.forwardPassNumber, model.stageNumber, 1))
    stopping_test::Bool = false
    iteration_count::Int64 = 0

    n = param.forwardPassNumber

    for i = 1:20
        stockTrajectories = forward_simulations(model,
                            param,
                            V,
                            n,
                            aleas)[2]
        backward_pass(model,
                      param,
                      V,
                      stockTrajectories,
                      aleas)
        # TODO: stopping test

        iteration_count+=1;
    end
end
