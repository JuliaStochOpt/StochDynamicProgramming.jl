#############################################################################
#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

type ActiveCutsContainer
    numCuts::Int
    territories::Array{Array} #set of states where cut k is active
    nstates::Int
    states::Array{Float64, 2} #set of states where cuts are tested
end


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
                    territory::Union{Array{Void}, Array{ActiveCutsContainer}},
                    it::Int64,
                    verbose::Int64)
    # Basic pruning: remove redundant cuts
    remove_redundant_cuts!(V)

    # If pruning is performed with territory heuristic, update territory
    # at given iteration:
    if isa(param.pruning[:type], Union{Type{Territory}, Type{LevelOne}})
        for t in 1:model.stageNumber-1
            states = reshape(trajectories[t, :, :], param.forwardPassNumber, model.dimStates)
            find_level1_cuts!(territory[t], V[t], states)
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


"""
Remove useless cuts in PolyhedralFunction.

# Arguments
* `model::SPModel`:
* `param::SDDPparameters`:
* `V::PolyhedralFunction`:
    Polyhedral function where cuts will be removed

# Return
* `PolyhedralFunction`: pruned polyhedral function
"""
function exact_cuts_pruning(model::SPModel, param::SDDPparameters, V::PolyhedralFunction, territory)
    ncuts = V.numCuts
    # Find all active cuts:
    if ncuts > 1
        active_cuts = Bool[is_cut_relevant(model, i, V, param.SOLVER)[1] for i=1:ncuts]
        return PolyhedralFunction(V.betas[active_cuts], V.lambdas[active_cuts, :], sum(active_cuts))
    else
        return V
    end
end


"""
Test whether the cut number k is relevant to define polyhedral function Vt.

# Arguments
* `model::SPModel`:
* `k::Int`:
    Position of cut to test in PolyhedralFunction object
* `Vt::PolyhedralFunction`:
    Object storing all cuts
* `solver`:
    Solver to use to solve linear problem
* `epsilon::Float64`: default is `1e-5`
    Acceptable tolerance to test cuts relevantness

# Return
* `Bool`: true if the cut is useful in the definition, false otherwise
"""
function is_cut_relevant(model::SPModel, k::Int, Vt::PolyhedralFunction, solver; epsilon=1e-5)
    m = Model(solver=solver)
    @variable(m, alpha)
    @variable(m, model.xlim[i][1] <= x[i=1:model.dimStates] <= model.xlim[i][2])

    for i in 1:Vt.numCuts
        if i!=k
            lambda = vec(Vt.lambdas[i, :])
            @constraint(m, Vt.betas[i] + dot(lambda, x) <= alpha)
        end
    end

    λ_k = vec(Vt.lambdas[k, :])
    @objective(m, Min, alpha - dot(λ_k, x) - Vt.betas[k])
    solve(m)
    sol = getobjectivevalue(m)
    return (sol < epsilon), getvalue(x)
end


########################################
# Territory algorithm
########################################

ActiveCutsContainer(ndim) = ActiveCutsContainer(0, [], 0, Array{Float64}(0, ndim))


"""Update territories (i.e. the set of tested states where
    a given cut is active) with cuts previously computed during backward pass.

# Arguments
* `cutscontainer::ActiveCutsContainer`:
* `Vt::PolyhedralFunction`:
    Object storing all cuts
* `states`:
    Object storing all visited states
"""
function find_level1_cuts!(cutscontainer::ActiveCutsContainer, V::PolyhedralFunction, states)
    nc = V.numCuts
    # get number of new positions to analyse:
    nx = size(states, 1)
    nt = nc - cutscontainer.numCuts

    for i in 1:nt
        add_cut!(cutscontainer)
        update_territory!(cutscontainer, V, nc - nt + i)
    end

    # ensure that territory has the same number of cuts as V!
    assert(cutscontainer.numCuts == V.numCuts)

    for i in 1:nx
        x = collect(states[i, :])
        add_state!(cutscontainer, V, x)
    end
end


"""Update territories (i.e. the set of tested states where
    a given cut is active) considering new cut given by index `indcut`.

# Arguments
* `cutscontainer::ActiveCutsContainer`:
* `V::PolyhedralFunction`:
    Object storing all cuts
* `indcut::Int64`:
    new cut index
"""
function update_territory!(cutscontainer::ActiveCutsContainer, V::PolyhedralFunction, indcut::Int64)
    for k in 1:cutscontainer.numCuts
        if k == indcut
            continue
        end
        todelete = []
        for (num, (ix, cost)) in enumerate(cutscontainer.territories[k])
            x = collect(cutscontainer.states[ix, :])

            costnewcut = cutvalue(V, indcut, x)

            if costnewcut > cost
                push!(todelete, num)
                push!(cutscontainer.territories[indcut], (ix, costnewcut))
            end
        end
        deleteat!(cutscontainer.territories[k], todelete)
    end
end


"""Add cut to `ActiveCutsContainer`."""
function add_cut!(cutscontainer::ActiveCutsContainer)
    push!(cutscontainer.territories, [])
    cutscontainer.numCuts += 1
end


"""Add a new state to test and accordingly update territories of each cut."""
function add_state!(cutscontainer::ActiveCutsContainer, V::PolyhedralFunction, x::Array{Float64})
    # Get cut which is active at point `x`:
    bcost, bcuts = optimalcut(x, V)

    # Add `x` to the territory of cut `bcuts`:
    cutscontainer.nstates += 1
    push!(cutscontainer.territories[bcuts], (cutscontainer.nstates, bcost))

    # Add `x` to the list of visited state:
    cutscontainer.states = vcat(cutscontainer.states, x')
end


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
                              V::PolyhedralFunction, cutscontainer::ActiveCutsContainer)
    assert(cutscontainer.numCuts == V.numCuts)

    nstates = [length(terr) for terr in cutscontainer.territories]
    active_cuts = nstates .> 0

    cutscontainer.territories = cutscontainer.territories[active_cuts]
    cutscontainer.numCuts = sum(active_cuts)
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
                                         cutscontainer::ActiveCutsContainer)

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


"""Find active cut at point `xf`.

# Arguments
* `xf::Vector{Float64}`:
* `V::PolyhedralFunction`:
    Object storing all cuts

# Return
`bestcost::Float64`
    Value of maximal cut at point `xf`
`bestcut::Int64`
    Index of maximal cut at point `xf`
"""
function optimalcut(xf::Vector{Float64}, V::PolyhedralFunction)
    bestcost = -Inf::Float64
    bestcut = -1
    nstates = size(V.lambdas, 2)
    ncuts = size(V.lambdas, 1)

    @inbounds for i in 1:ncuts
        cost = V.betas[i]
        for j in 1:nstates
            cost += V.lambdas[i, j]*xf[j]
        end
        if cost > bestcost
            bestcost = cost
            bestcut = i
        end
    end
    return bestcost, bestcut
end


"""
Get value of cut with index `indc` at point `x`.

# Arguments
- `V::PolyhedralFunction`
    Approximation of the value function as linear cuts
- `indc::Int64`
    Index of cut to consider
- `x::Array{Float64}`
    Coordinates of state

# Return
`cost::Float64`
    Value of cut `indc` at point `x`
"""
function cutvalue(V::PolyhedralFunction, indc::Int, x::Array{Float64})
    cost = V.betas[indc]
    for j in 1:size(V.lambdas, 2)
        cost += V.lambdas[indc, j]*x[j]
    end
    return cost
end

# Return total number of cuts in PolyhedralFunction array:
get_total_number_cuts(V::Array{PolyhedralFunction}) = sum([v.numCuts for v in V])

