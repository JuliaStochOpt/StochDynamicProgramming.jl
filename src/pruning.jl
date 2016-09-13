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

Territories(ndim) = Territories(0, [], 0, Array{Float64}(0, ndim))


"""
Exact pruning of all polyhedral functions in input array.

# Arguments
* `model::SPModel`:
* `params::SDDPparameters`:
* `Vector{PolyhedralFunction}`:
    Polyhedral functions where cuts will be removed
"""
function prune_cuts!(model::SPModel, params::SDDPparameters, V::Vector{PolyhedralFunction})
    for i in 1:length(V)-1
        V[i] = exact_prune_cuts(model, params, V[i])
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
function territory_prune_cuts(territory::Territories,
                              V::PolyhedralFunction,
                              states::Array{Float64, 2})
    find_territory!(territory, V, states)
    return remove_empty_cuts(territory, V)
end


""" Update territories with cuts previously computed during backward pass.  """
function find_territory!(territory, V, states)
    nc = V.numCuts
    # get number of new positions to analyse:
    nx = size(states, 1)

    for i in 1:V.numCuts
        add_cut!(territory, V)
        update_territory!(territory, V, nc - nx + i)
    end
    #= assert(territory.ncuts == V.numCuts) =#

    for i in 1:nx
        x = collect(states[i, :])
        add_state!(territory, V, x)
    end

end


""" Update territories with new cuts added during previous backward pass.  """
function update_territory!(territory, V, indcut)
    for k in 1:territory.ncuts
        if k == indcut
            continue
        end
        for (num, (ix, cost)) in enumerate(territory.territories[k])
            x = collect(territory.states[ix, :])

            costnewcut = cutvalue(V, indcut, x)

            if costnewcut > cost
                deleteat!(territory.territories[k], num)
                push!(territory.territories[indcut], (ix, costnewcut))
            end
        end
    end
end


"""Add a new cut to the"""
function add_cut!(territory, V)
    push!(territory.territories, [])
    territory.ncuts += 1
end


function add_state!(territory, V, x)
    bcost, bcuts = optimalcut(x, V)

    indx = territory.nstates + 1
    push!(territory.territories[bcuts], (indx, bcost))

    territory.states = vcat(territory.states, x')
    territory.nstates += 1
end


"""
Remove empty cuts in PolyhedralFunction
"""
function remove_empty_cuts!(territory, V)
    assert(length(territory) == V.numCuts)

    nstates = [length(cont) for cont in territory.territories]
    active_cuts = nstates .> 0

    territory.territories = territory.territories[active_cuts]
    territory.ncuts = sum(active_cuts)
    return PolyhedralFunction(V.betas[active_cuts],
                              V.lambdas[active_cuts, :],
                              sum(active_cuts))
end


"""
Get cut which approximate the best value function at point `x`.
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


""" Get approximation of value function at given point `x`.  """
function cutvalue(V, indc, x)
    cost = V.betas[indc]
    for j in 1:size(V.lambdas, 2)
        cost += V.lambdas[indc, j]*xf[j]
    end
    return cost
end

