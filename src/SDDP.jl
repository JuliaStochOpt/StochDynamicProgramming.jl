#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# SDDP is an implementation of the Stochastic Dual Dynamic Programming 
# algorithm for multi-stage stochastic convex optimization problem
# see TODO
#############################################################################

#module SDDP

import JuMP #TODO : require JuMP ?

#export #TODO
#Objects SPModel
    
include("utility.jl")
include("oneStepOneAleaProblem.jl")
include("forwardBackwardIterations.jl")
include("SDDPoptimize.jl")

type NoiseLaw
    supportSize::Int16 
    support::Array{AbstractFloat,2}
    proba::Tuple{Float16}
end

abstract SPModel
    
type DeterministSPModel <: SPModel
    #problemDimension
    stageNumber
    dimControls
    dimStates
    initialState
    costFunctions # TODO collection of cost function
    dynamics # TODO collection of dynamic function
    #noises::Vector{NoiseLaw} # TODO collection of noises law
end

type RandomSPModel <: SPModel
    #problemDimension
    stageNumber
    dimControls
    dimStates
    initialState
    costFunctions # TODO collection of cost function
    dynamics # TODO collection of dynamic function
    noises::Vector{NoiseLaw} # TODO collection of noises law
end

#type LinearDynamicLinearCostSPmodel <: SPModel
#    problemDimension
#    stageNumber#::int
#    dimControls
#    dimStates
#    
#    initialState
#    
#    costFunctions # TODO collection of cost function
#    dynamics # TODO collection of dynamic function
#    noises::Vector{NoiseLaw} # TODO collection of noises law
#end




type SDDPparameters
    solver#::MathProgBase. #TODO
    forwardPassNumber#::int # number of simulated scenario in the forward pass
    initialization #TODO 
    stoppingTest #TODO
    
end

type PolyhedralFunction
    #function defined by max_k betas[k] + lambdas[k,:]*x
    betas::Vector{Float64}
    lambdas::Array{Float64,2} #lambdas[k,:] is the subgradient
end

test = DeterministSPModel(3,[2;2;2],[2;2;2],[1;1],costFunction,dynamic);

cut1 = PolyhedralFunction([1;1],[1 0;0 1]);
cut2 = PolyhedralFunction([2;2],[2 0;0 2]);
cut3 = PolyhedralFunction([3;3],[3 0;0 3]);

cut = Any[cut1,cut2,cut3];

stockTrajectories = Any[[1;1],[2;2],[3;3]];

#end
