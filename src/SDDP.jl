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

using MathProgBase #TODO interface with JuMP
using GLPKMathProgInterface
using CPLEX

#export #TODO
#Objects SPModel
    

include("simulate.jl");

println("NoiseLaw Type defined");

abstract SPModel
    
type LinearDynamicLinearCostSPmodel <: SPModel
    #problemDimension
    stageNumber::Int64
    dimControls::Array{Int64,1}
    dimStates::Array{Int64,1}
    initialState
    costFunctions#::Tuple{Vector{Float64}}
    dynamics#::Tuple{Array{Float64,2}}
    #constraints#::Tuple{Array{Float64,2}}
    lowerbounds#::Tuple{Vector{Float64}}
    upperbounds#::Tuple{Vector{Float64}}
    noises#::Vector{NoiseLaw} # TODO collection of noises law
end
println("SPModel Type defined");

type SDDPparameters
    solver#::MathProgBase #TODO
    forwardPassNumber::Int64#::int # number of simulated scenario in the forward pass
    initialization #TODO 
    stoppingTest #TODO 
end
println("SDDPparameters Type defined");

type PolyhedralFunction
    #function defined by max_k betas[k] + lambdas[k,:]*x
    betas::Vector{Float64}
    lambdas::Array{Float64,2} #lambdas[k,:] is the subgradient
end
println("Polyhedral Type defined");

println(" ");
println("All types well defined")
println(" ");

include("oneStepOneAleaProblem.jl")
println("oneStepOneAleaProblem.jl file included");

#include("utility.jl")
#println("utility.jl file included");

include("forwardBackwardIterations.jl")
println("forwardBackwardIterations.jl file included");

include("SDDPoptimize.jl")
println("SDDPoptimize.jl file included");

#end
