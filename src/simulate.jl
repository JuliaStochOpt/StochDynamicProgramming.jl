#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Probability utilities
#############################################################################


#TODO _const should not exist 

using Distributions

type NoiseLaw
    supportSize::UInt16 
    support::Array{Float64,2}#TODO risque de poser des problèmes
    proba::Vector{Float64}
end

function NoiseLaw_const(supportSize,support,proba)
    supportSize = convert(UInt16,supportSize)
    if ndims(support)==1
        support = reshape(support,1,length(support))
    end
    
    if ndims(proba) == 2
        proba = vec(proba)
    elseif  ndims(proba) >= 2
        proba = squeeze(proba,1)
    end    
    
    if sum(proba) !=1 error("probability doesnot sum to 1") end
    
    return NoiseLaw(supportSize,support,proba) 
end

function NoiseLaw(support,proba)
    return NoiseLaw_const(length(proba),support,proba)
end




"""
Simulate n scenario according to a given NoiseLaw


Parameters:
- law::Vector{NoiseLaw}
    Vector of discrete independent random variables
    
- n::Int
    number of simulations computed

Returns :
- scenarios Array(Float32,n,T)
    an Array of scenario, scenarios[i,:] being the ith noise scenario
"""

function simulate(law::Vector{NoiseLaw},n::Int)
    if n <= 0 
        error("negative number of simulations")
    end
    Tf = length(law)
    scenarios = Array{Float32}(n,Tf)
    for i = 1:n#TODO can be parallelized
        scenario = []
        for t=1:Tf
            new_val = law[t].support[rand(Categorical(law[t].proba))]
            push!(scenario, new_val)
        end
        scenarios[i,:]=scenario
    end

    return scenarios
end

### test
supp = [1, 2, 3] 
p = [0.1 0.4 0.5] 
w1 = NoiseLaw_const(3,supp,p)
w2 = NoiseLaw(supp,p)
simulate([w1,w2],3)


