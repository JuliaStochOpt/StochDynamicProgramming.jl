#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# run unit-tests
#############################################################################


using StochDynamicProgramming
using Clp, JuMP, Nullables
using Printf
using Statistics
using Test
using CutPruners

# Test utility functions:
include("prob.jl")

# Test risk measures:
include("changeprob.jl")

# Test SDDP:
# Use Gurobi solver or something Scalar Quadratic Function support Solver
# Clp do not support Quadratic
include("sddp.jl")

# Test DP:
include("sdp.jl")

# Test extensive formulation:
include("extensive_formulation.jl")
