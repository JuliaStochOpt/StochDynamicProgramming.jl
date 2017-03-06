#############################################################################
#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################



"""
Exact pruning of all polyhedral functions in input array.

# Arguments
* `model::SPModel`:
* `trajectories::Array{Float64, 3}`
    Previous trajectories
"""
function prune!(sddp::SDDPInterface,
                trajectories::Array{Float64, 3},
                )
    # Basic pruning: remove redundant cuts
    #= for t in 1:sddp.spmodel.stageNumber-1 =#
    #=     b, A = fetchnewcuts!(sddp.bellmanfunctions[t]) =#
    #=     nnew = length(b) =#
    #=     if nnew > 0 =#
    #=         mycut = Bool[true for _ in 1:length(b)] =#
    #=         CutPruners.addcuts!(sddp.pruner[t], A, b, mycut) =#
    #=     end =#
    #= end =#

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

