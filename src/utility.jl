#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Some useful methods
#
#############################################################################


function cost_function(t,x,u,xi)
    #TODO
    return cost
end

function dynamic(t,x,u,xi)
    #TODO
    return new_state
end




"""
Estimate the upper bound with the Monte-Carlo error


Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm
    
- V (bellmanFunctions)
    the current estimation of Bellman's functions 
    
- forwardPassNumber (int)
    number of Monte-Carlo simulation
    
- returnMCerror (Bool)
    return or not the estimation of the MC error
    

Returns :
- estimated-upper bound
- Monte-Carlo error on the upper bound (if returnMCerror)

"""
function upper_bound(model::SPmodel,
                     param::SDDPparameters,
                     V::Vector{PolyhedralFunction},
                     forwardPassNumber::Int64,
                     returnMCerror::Bool)

    C = forward_simulations(model,param,V,forwardPassNumber,nothing,true, false, false);
    m = mean(C) 
    if returnMCerror
        return m, 1.96*std(C)/sqrt(forwardPassNumber)
    else 
        return m
    end
end
