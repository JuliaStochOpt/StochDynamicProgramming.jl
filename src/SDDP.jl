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

using JuMP

export optimize, NoiseLaw, simulate_scenarios,
        SDDPparameters, LinearDynamicLinearCostSPmodel,
        PolyhedralFunction, NextStep, forward_simulations

include("simulate.jl")
include("SDDPoptimize.jl")
include("forwardBackwardIterations.jl")
include("objects.jl")


end
