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
construct a NoiseLaw from the support and associated proba

Parameters:
- support (Array{Float})
    an array of the support of the noise
- proba (Tuple{Float})
    an array of the proba associated to the support. should be of the same size as support
    and sum to 1.
    
Returns
- the NoiseLaw
"""
function generateLaw(support,probas)
    
    if ndims(support) ==1
        #TODO le passer à 2
    elseif ndims(support)==2
        n = size(support)[2]
    else
        error("the support is an array of dimension greater than 2")
    end    
    
    
    if length(probas)!=n
        error("size of support does not match the number of probability weight")
    end
    if sum(a)!=1
        error("probability weights does not sum to 1")
    end
    
    convert(Int16,n)
    convert(Array{AbstractFloat,2},support)
    convert(Tuple{Float16},collect(probas))
    
    return NoiseLaw(n,support,probas)
end

function simulate(law::NoiseLaw,n::Int)
    
end



