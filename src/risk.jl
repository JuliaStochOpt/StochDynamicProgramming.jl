#  Copyright 2017, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Implement different functions to deal with risk in SDDP
#############################################################################

"""
Risk measures consider here can be written (in case of a minimization problem)
   sup     E_{p}[x]
  p in P
where P is a convex set of probability distribution.
The function returns the argsup allowing to compute the
risk measure as a expectation

$(SIGNATURES)

# Description
Return the probability among the set of possible probabilities
allowing to compute the risk

# Arguments
* `prob::Array{float,1}`:
    probability distribution
* `riskMeasure::RiskMeasure`:
    risk Measure
* `costs`:
    array of costs

# Returns
* `proba::Array{float,1}`:
    a new probability distribution taking risk into account.

"""
function risk_proba(prob, riskMeasure::RiskMeasure,costs)
    error("'risk_proba' not defined for $(typedof(s))")
end

"""
$(TYPEDEF)

Return the probability distribution to compute a Average Value at Risk of level beta.
"""
function risk_proba(prob,riskMeasure::AVaR,costs)
    perm = sortperm(costs,rev = true)
    beta = riskMeasure.beta
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

"""
$(TYPEDEF)

Leave the probability distribution unchanged to compute the expectation.
"""
function risk_proba(prob,riskMeasure::Expectation,costs)
    return prob
end

"""
$(TYPEDEF)
perm
Return a dirac on the worst cost as a probability distribution.
"""
function risk_proba(prob,riskMeasure::WorstCase,costs)
    proba = zeros(length(prob))
    proba[indmax(costs)] = 1
    return proba
end

"""
$(TYPEDEF)

Return the probability distribution to compute a convex combination
between expactation and an Average Value at Risk of level beta.
"""
function risk_proba(prob,riskMeasure::ConvexCombi,costs)
    perm = sortperm(costs,rev = true)
    beta = riskMeasure.beta
    lambda = riskMeasure.lambda
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
    return lambda*prob + (1-lambda)*aux
end

"""
$(TYPEDEF)

Return the worst extreme probability distribution
defining the convex set P
"""
function risk_proba(prob,riskMeasure::PolyhedralRisk,costs)
    P = riskMeasure.polyset
    valuesup = P*costs
    return P[indmax(valuesup),:]
end
