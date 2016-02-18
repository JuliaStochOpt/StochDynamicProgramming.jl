#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# run unit-tests
#############################################################################

push!(LOAD_PATH, "src")

using StochDynamicProgramming
using Distributions
using FactCheck


# Test simulate.jl
facts("Probability functions") do
    support = [1, 2, 3]
    proba = [.2 .5 .3]

    law = NoiseLaw(support, proba)
    @fact typeof(law) --> NoiseLaw
    @fact law.supportSize --> 3

    dims = (2, 2, 1)
    scenarios = simulate_scenarios([law, law], dims)
    @fact typeof(scenarios) --> Array{Float64, 3}
    @fact size(scenarios) --> (2, 2, 1)

    scenarios2 = simulate_scenarios(Normal(), dims)
    @fact typeof(scenarios2) --> Array{Float64, 3}
end


facts("Utility functions") do
    V = StochDynamicProgramming.get_null_value_functions()
    @fact typeof(V) --> PolyhedralFunction
    @fact V.betas[1] --> 0

    arr = rand(4, 4, 2)
    vec = StochDynamicProgramming.extract_vector_from_3Dmatrix(arr, 2, 1)
    @fact typeof(vec) --> Vector{Float64}
    @fact size(vec) --> (2,)
end


facts("Dam management") do
    include("../examples/dam.jl")
    solve_dams()
end
