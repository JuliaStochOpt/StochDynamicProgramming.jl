#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  utility functions
#
#############################################################################

import Base: +, show

"""
Write Polyhedral functions in a CSV file.

# Arguments
* `filename::String`:
    Name of output filt
* `Vts::Vector{PolyhedralFunction}`:
    Vector of polyhedral functions to save
"""
function writecsv(filename::AbstractString, Vts::Vector{PolyhedralFunction})
    outfile = open(filename, "w")

    time = 1
    for V in Vts
        ncuts = V.numCuts

        for i in 1:ncuts
            write(outfile, join((time, V.betas[i], tuple(V.lambdas[i, :]...)...), ","), "\n")
        end

        time += 1
    end

    close(outfile)
end


"""
Import cuts from a dump text file.

# Argument
* `dump::String`:
    Name of file to import
# Return
`Vector{PolyhedralFunction}`
"""
function read_polyhedral_functions(dump::AbstractString)
    # Store all values in a two dimensional array:
    process = readdlm(dump, ',')

    ntime = round(Int, maximum(process[:, 1]))
    V = Vector{PolyhedralFunction}(ntime)
    total_cuts = size(process)[1]
    dim_state = size(process)[2] - 2

    for it in 1:total_cuts
        # Read time corresponding to cuts:
        t = round(Int, process[it, 1])
        beta = process[it, 2]
        lambda = vec(process[it, 3:end])

        try
            V[t].lambdas = vcat(V[t].lambdas, lambda')
            V[t].betas = vcat(V[t].betas, beta)
            V[t].numCuts += 1
        catch
            V[t] = PolyhedralFunction([beta],
                                       reshape(lambda, 1, dim_state), 1)
        end
    end
    return V
end


"""Concatenate collection of arrays of PolyhedralFunction."""
function catcutsarray(polyfunarray::Vector{StochDynamicProgramming.PolyhedralFunction}...)
    assert(length(polyfunarray) > 0)
    ntimes = length(polyfunarray[1])
    # Concatenate cuts in polyfunarray, and discard final time as we do not add cuts at final time:
    concatcuts = StochDynamicProgramming.PolyhedralFunction[catcuts([V[t] for V in polyfunarray]...) for t in 1:ntimes-1]
    return vcat(concatcuts, polyfunarray[1][end])
end


"""Concatenate collection of PolyhedralFunction."""
function catcuts(Vts::StochDynamicProgramming.PolyhedralFunction...)
    betas = vcat([V.betas for V in Vts]...)
    lambdas = vcat([V.lambdas for V in Vts]...)
    numcuts = sum([V.numCuts for V in Vts])
    return StochDynamicProgramming.PolyhedralFunction(betas, lambdas, numcuts)
end


"""
Generate a random state.

# Arguments
* `model::SPModel`:

# Return
`Vector{Float64}`, shape=`(model.dimStates,)`

"""
function get_random_state(model::SPModel)
    return [model.xlim[i][1] + rand()*(model.xlim[i][2] - model.xlim[i][1]) for i in 1:model.dimStates]
end


"""
Print in stdout:
Pass number     Upper bound     Lower bound     exectime

# Arguments
* `io::IO`:
* `stats::SDDPStat`:

"""
function Base.show(io::IO, stats::SDDPStat)
    print("Pass n\° ", stats.niterations)
    (stats.upper_bounds[end] < Inf) && @printf("\tUpper-bound: %.4e", stats.upper_bounds[end])
    @printf("\tLower-bound: %.4e", stats.lower_bounds[end])
    print("\tTime: ", round(stats.exectime[end], 2),"s")
end


