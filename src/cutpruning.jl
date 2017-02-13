#############################################################################
#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################



""" Remove redundant cuts in Polyhedral Value function `V`

# Arguments
* `V::PolyhedralFunction`:
"""
function remove_cuts!(V::PolyhedralFunction)
    Vf = hcat(V.lambdas, V.betas)
    Vf = unique(Vf, 1)
    return PolyhedralFunction(Vf[:, end], Vf[:, 1:end-1], size(Vf)[1])
end


""" Remove redundant cuts in a vector of Polyhedral Functions `Vts`.

# Arguments
* `Vts::Vector{PolyhedralFunction}`:
"""
function remove_redundant_cuts!(Vts::Vector{PolyhedralFunction})
    n_functions = length(Vts)-1
    for i in 1:n_functions
        Vts[i] = remove_cuts!(Vts[i])
    end
end


"""
Exact pruning of all polyhedral functions in input array.

# Arguments
* `model::SPModel`:
* `param::SDDPparameters`:
* `Vector{PolyhedralFunction}`:
    Polyhedral functions where cuts will be removed
* `trajectories::Array{Float64, 3}`
    Previous trajectories
* `territory::Array{ActiveCutsContainer}`
    Container storing the territory (i.e. the set of tested states where
    a given cut is active) for each cuts
* `it::Int64`:
    current iteration number
* `verbose::Int64`
"""
function prune_cuts!(model::SPModel,
                    param::SDDPparameters,
                    V::Vector{PolyhedralFunction},
                    trajectories::Array{Float64, 3},
                    territory,
                    it::Int64,
                    verbose::Int64)
    # Basic pruning: remove redundant cuts
    remove_redundant_cuts!(V)

    # If pruning is performed with territory heuristic, update territory
    # at given iteration:
    if isa(param.pruning[:type], Union{Type{Territory}, Type{LevelOne}})
        for t in 1:model.stageNumber-1
            states = reshape(trajectories[t, :, :], param.forwardPassNumber, model.dimStates)
            # FIXME: add in a proper way new cuts in territory
            territory[t].cuts_DE = V[t].lambdas
            territory[t].cuts_de = V[t].betas
            CutPruners.updatestats!(territory[t], states)
        end
    end

    # If specified to prune cuts at this iteration, do it:
    if param.pruning[:pruning] && (it%param.pruning[:period]==0)
        # initial number of cuts:
        ncuts_initial = get_total_number_cuts(V)
        (verbose > 0) && print("Prune cuts ...")

        for i in 1:length(V)-1
            V[i] = pcuts!(param.pruning[:type], model, param, V[i], territory[i])
        end

        # final number of cuts:
        ncuts_final = get_total_number_cuts(V)

        (verbose > 0) && @printf(" New cuts/Old cuts ratio: %.3f \n", ncuts_final/ncuts_initial)
    end
end

pcuts!(::Type{LevelOne}, model, param, V, territory) = level1_cuts_pruning!(model, param, V, territory)
pcuts!(::Type{ExactPruning}, model, param, V, territory) = exact_cuts_pruning(model, param, V, territory)
pcuts!(::Type{Territory}, model, param, V, territory) = exact_cuts_pruning_accelerated!(model, param, V, territory)



"""Remove cuts in PolyhedralFunction that are inactive on all visited states.
# Arguments
* `cutscontainer::ActiveCutsContainer`:
* `V::PolyhedralFunction`:
    Object storing all cuts
# Return
* `V::PolyhedralFunction`
    the new PolyhedralFunction
"""
function level1_cuts_pruning!(model::SPModel, param::SDDPparameters,
                              V::PolyhedralFunction, cutscontainer::CutPruners.LevelOneCutPruner)

    nstates = [length(terr) for terr in cutscontainer.territories]
    active_cuts = nstates .> 0

    cutscontainer.territories = cutscontainer.territories[active_cuts]
    return PolyhedralFunction(V.betas[active_cuts],
                              V.lambdas[active_cuts, :],
                              sum(active_cuts))
end


"""Remove useless cuts in PolyhedralFunction

First test if cuts are active on the visited states,
then test remaining cuts.

# Arguments
* `model::SPModel`:
* `cutscontainer::ActiveCutsContainer`:
* `V::PolyhedralFunction`:
    Object storing all cuts

# Return
* `V::PolyhedralFunction`
    the new PolyhedralFunction
"""
function exact_cuts_pruning_accelerated!(model::SPModel, param::SDDPparameters,
                                         V::PolyhedralFunction,
                                         cutscontainer::CutPruners.LevelOneCutPruner)

    assert(cutscontainer.numCuts == V.numCuts)
    solver = param.SOLVER

    nstates = [length(terr) for terr in cutscontainer.territories]
    # Set of inactive cuts:
    inactive_cuts = nstates .== 0
    # Set of active cuts:
    active_cuts = nstates .> 0

    # get index of inactive cuts:
    index = collect(1:cutscontainer.numCuts)[inactive_cuts]

    # Check if inactive cuts are useful or not:
    for id in index
        status, x = is_cut_relevant(model, id, V, solver)
        if status
            active_cuts[id] = true
        end
    end

    # Remove useless cuts:
    cutscontainer.territories = cutscontainer.territories[active_cuts]
    cutscontainer.numCuts = sum(active_cuts)
    return PolyhedralFunction(V.betas[active_cuts],
                              V.lambdas[active_cuts, :],
                              sum(active_cuts))
end


# Return total number of cuts in PolyhedralFunction array:
ncuts(V::PolyhedralFunction) = length(V.betas)
ncuts(V::Array{PolyhedralFunction}) = sum([ncuts(v) for v in V])

