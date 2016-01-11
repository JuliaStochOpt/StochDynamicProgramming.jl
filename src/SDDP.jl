#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# SDDP is an implementation of the Stochastic Dual Dynamic Programming
# algorithm for multi-stage stochastic convex optimization problem
# see TODO
#############################################################################

module SDDP

import JuMP #TODO : require JuMP ?
import Distibutions

export TODO
#Objects
    SPModel

include("utils.jl")
include("oneStepOneAleaProblem.jl")
include("forwardBackwardIterations.jl")
include("SDDPoptimize.jl")

abstract SPModel


type LinearDynamicLinearCostSPmodel :< SPModel
    # problem dimension
    stageNumber::Int64
    dimControls::Int64
    dimStates::Int64

    initialState

    costFunctions # TODO collection of cost function
    dynamics # TODO collection of dynamic function
    noises::Vector{NoiseLaw} # TODO collection of noises law
end



type SDDPparameters
    solver::MathProgBase #TODO
    forwardPassNumber::Int64 # number of simulated scenario in the forward pass
    initialization #TODO
    maxItNumber #TODO
end

type PolyhedralFunction
    #function defined by max_k betas[k] + lambdas[k,:]*x
    betas::Vector{Float64}
    lambdas::Array{Float64,2} #lambdas[k,:] is the subgradient

    # number of cuts:
    numCuts::Int64
end


end
