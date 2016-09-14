#############################################################################
#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

type Territories
    ncuts::Int
    territories::Array{Array}
    nstates::Int
    states::Array{Float64, 2}
end


"""
Exact pruning of all polyhedral functions in input array.

# Arguments
* `model::SPModel`:
* `params::SDDPparameters`:
* `Vector{PolyhedralFunction}`:
    Polyhedral functions where cuts will be removed
* `trajectories::Array{Float64, 3}`
    Previous trajectories
* `territory::Array{Territories}`
    Container storing the territory for each cuts
* `it::Int64`:
    current iteration number
* `verbose::Int64`
"""
function prune_cuts!(model::SPModel,
                    param::SDDPparameters,
                    V::Vector{PolyhedralFunction},
                    trajectories::Array{Float64, 3},
                    territory::Union{Void, Array{Territories}},
                    it::Int64,
                    verbose::Int64)
    # Basic pruning: remove redundant cuts
    remove_redundant_cuts!(V)

    # If pruning is performed with territory heuristic, update territory
    # at given iteration:
    if param.pruning[:type] == "territory"
        for t in 1:model.stageNumber-1
            states = reshape(trajectories[t, :, :], param.forwardPassNumber, model.dimStates)
            find_territory!(territory[t], V[t], states)
        end
    end

    # If specified to prune cuts at this iteration, do it:
    if param.pruning[:pruning] && (it%param.pruning[:period]==0)
        # initial number of cuts:
        ncuts_initial = get_total_number_cuts(V)
        (verbose > 0) && print("Prune cuts ...")

        for i in 1:length(V)-1
            if param.pruning[:type] == "exact"
                # apply exact cuts pruning:
                V[i] = exact_prune_cuts(model, param, V[i])
            elseif param.pruning[:type] == "territory"
                # apply heuristic to prune cuts:
                V[i] = remove_empty_cuts!(territory[i], V[i])
            end
        end

        # final number of cuts:
        ncuts_final = get_total_number_cuts(V)

        (verbose > 0) && println(" Deflation: ", ncuts_final/ncuts_initial)
    end
end


"""
Remove useless cuts in PolyhedralFunction.

# Arguments
* `model::SPModel`:
* `params::SDDPparameters`:
* `V::PolyhedralFunction`:
    Polyhedral function where cuts will be removed

# Return
* `PolyhedralFunction`: pruned polyhedral function
"""
function exact_prune_cuts(model::SPModel, params::SDDPparameters, V::PolyhedralFunction)
    ncuts = V.numCuts
    # Find all active cuts:
    if ncuts > 1
        active_cuts = Bool[is_cut_relevant(model, i, V, params.SOLVER) for i=1:ncuts]
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
    return sol < epsilon
end


########################################
# Territory algorithm
########################################

Territories(ndim) = Territories(0, [], 0, Array{Float64}(0, ndim))


""" Update territories with cuts previously computed during backward pass.  """
function find_territory!(territory, V, states)
    nc = V.numCuts
    # get number of new positions to analyse:
    nx = size(states, 1)
    nt = nc - territory.ncuts

    for i in 1:nt
        add_cut!(territory)
        update_territory!(territory, V, nc - nt + i)
    end

    # ensure that territory has the same number of cuts as V!
    assert(territory.ncuts == V.numCuts)

    for i in 1:nx
        x = collect(states[i, :])
        add_state!(territory, V, x)
    end

end


"""Update territories considering new cut with index `indcut`."""
function update_territory!(territory, V, indcut)
    for k in 1:territory.ncuts
        if k == indcut
            continue
        end
        todelete = []
        for (num, (ix, cost)) in enumerate(territory.territories[k])
            x = collect(territory.states[ix, :])

            costnewcut = cutvalue(V, indcut, x)

            if costnewcut > cost
                push!(todelete, num)
                push!(territory.territories[indcut], (ix, costnewcut))
            end
        end
        deleteat!(territory.territories[k], todelete)
    end
end


"""Add cut to `territory`."""
function add_cut!(territory)
    push!(territory.territories, [])
    territory.ncuts += 1
end


"""Add a new state and update territories."""
function add_state!(territory::Territories, V::PolyhedralFunction, x::Array{Float64})
    # Get cut which is the supremum at point `x`:
    bcost, bcuts = optimalcut(x, V)

    # Add `x` to the territory of cut `bcuts`:
    territory.nstates += 1
    push!(territory.territories[bcuts], (territory.nstates, bcost))

    # Add `x` to the list of visited state:
    territory.states = vcat(territory.states, x')
end


"""Remove empty cuts in PolyhedralFunction"""
function remove_empty_cuts!(territory::Territories, V::PolyhedralFunction)
    assert(territory.ncuts == V.numCuts)

    nstates = [length(terr) for terr in territory.territories]
    active_cuts = nstates .> 0

    territory.territories = territory.territories[active_cuts]
    territory.ncuts = sum(active_cuts)
    return PolyhedralFunction(V.betas[active_cuts],
                              V.lambdas[active_cuts, :],
                              sum(active_cuts))
end


"""Get cut which approximate the best value function at point `x`."""
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
Get approximation of value function at given point `x`.

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

