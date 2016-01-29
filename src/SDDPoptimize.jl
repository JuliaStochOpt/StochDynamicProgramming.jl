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

"""
TODO: add docstring
TODO: move initialize in proper module
TODO: fix initialize

"""
function initialize_value_functions_array(model::SDDP.SPModel)

    V = Vector{SDDP.PolyhedralFunction}(model.stageNumber)
    for t = 1:model.stageNumber
        V[t] = initialize_value_functions()
    end

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
                  param::SDDP.SDDPparameters)

    # Initialize value functions:
    V = initialize_value_functions_array(model)
    aleas = rand(param.forwardPassNumber, model.stageNumber, 1)
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