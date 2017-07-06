#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Implement different functions to deal with risk in SDDP
#############################################################################

"""
Update probability distribution to consider only worst case scenarios

$(SIGNATURES)

# Description
Keep only the tail of the probability distribution given an order
and renormalized it

# Arguments
* `prob::probabilitydistribution`:
    SDDP interface object
* `beta::Float64`:
    real number
* `perm::Array{float,1}`:
    array of permutations

# Returns
* `aux::Array{float,1}`:
    a new probability distribution taking risk into account.

"""
function change_proba_risk(prob,beta,perm)
    proba = zeros(length(prob))
    for i in 1:length(perm)
        proba[i] = prob[perm[i]]
    end
    index = findfirst(x -> x>=beta ,cumsum(proba))
    if index > 1
        proba[index] = (beta - cumsum(proba)[index - 1])
    end
    for i in (index + 1):length(proba)
        proba[i] = 0
    end
    proba = proba / sum(proba)
    aux = zeros(length(prob))
    for i in 1:length(perm)
        aux[perm[i]] = proba[i]
    end 
    return aux
end
