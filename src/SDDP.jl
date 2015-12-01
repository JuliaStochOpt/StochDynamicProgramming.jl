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

export TODO
#Objects
    SPModel
    
include("utils.jl")
include("oneStepOneAleaProblem.jl")
include("forwardBackwardIterations.jl")
include("SDDPoptimize.jl")

type SPModel 
    SPModelType::str # TODO specify the model type (structure of costs, etc) among a given list
    
    # problem dimension
    stageNumber::int
    dimControls
    dimStates
    
    initialState
    
    costFunctions # TODO collection of cost function
    dynamics # TODO collection of dynamic function
    noises # TODO collection of noises law
end

function SPModel()
#TODO standard constructor
end


type SDDPinstance
    model::SPModel
    solver::MathProgBase. #TODO
    
    valueFunctionsApprox::Vector{polyhedralFunction}
    
    forwardPassNumber::int # number of simulated scenario in the forward pass
    
    
    initialization #TODO 
    stoppingTest #TODO
    
end

type polyhedralFunction
# TODO collection of functions defined by cuts
end


end
