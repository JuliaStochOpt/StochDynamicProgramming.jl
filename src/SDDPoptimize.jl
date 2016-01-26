#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  the actual optimization function
#
#############################################################################



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


Returns :
- V::Array{PolyhedralFunction}
    the collection of approximation of the bellman functions

"""
function optimize(model::SDDP.SPModel,
                  param::SDDP.SDDPparameters)
    # TODO initialization (V and so on)

    stopping_test::Bool = false;
    iteration_count::int = 0;

    n = param.forwardPassNumber

    for i = 1:20
        stockTrajectories = forwardSimulations(model,
                            param,
                            V,
                            n,
                            returnCosts = false,
                            returnStocks=true,
                            returnControls= false);
        backwardPass(model,
                      param,
                      V,
                      stockTrajectories);
        #TODO stopping test

        iteration_count+=1;
    end
end
