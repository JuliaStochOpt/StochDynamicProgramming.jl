#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# run unit-tests
#############################################################################


using StochDynamicProgramming
using Distributions, Clp, FactCheck, JuMP


# Test utility functions:
include("prob.jl")
include("utils.jl")

# Test SDDP:
include("sddp.jl")

# Test DP:
include("sdp.jl")

# Test extensive formulation:
include("extensive_formulation.jl")
