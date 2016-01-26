#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  the actual optimization function 
#
#############################################################################

#=
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
=#

function optimize(model::LinearDynamicLinearCostSPmodel,
                  param::SDDPparameters,
                  V,xi)
    	# TODO initialization (V and so on)

	stopping_test = param.stoppingTest;
    	iteration_count = 0;
    
    	n = param.forwardPassNumber
    	#while(!stopping_test)
    	if (stopping_test[1] == 0)
         	while(iteration_count <stopping_test[2])
             auxiliaire = forward_simulations(model,
                                 param,
                                 V,
                                 n, 
                                 xi,# = nothing,
                                 false,  
                                 true, 
                                 false)
             
             stocks = auxiliaire.sto
             backward_pass(model,
                           param,
                           V,
                           stocks,
                           xi);
             #TODO stopping test
             
             iteration_count+=1;
             println("iteration = ",iteration_count);
          end
     end
end
