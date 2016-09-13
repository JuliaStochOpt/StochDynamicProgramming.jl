#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define the Forward / Backward iterations of the SDDP algorithm
#############################################################################


"""
Exact pruning of all polyhedral functions in input array.

# Arguments
* `model::SPModel`:
* `params::SDDPparameters`:
* `Vector{PolyhedralFunction}`:
    Polyhedral functions where cuts will be removed
* `iteration_count::Int64`:
    current iteration number
* `verbose::Int64`
"""
function prune_cuts!(model::SPModel, 
                    param::SDDPparameters, 
                    V::Vector{PolyhedralFunction},
                    iteration_count::Int64,
                    verbose::Int64)
    # If specified, prune cuts:
    if (param.compute_cuts_pruning > 0) && (iteration_count%param.compute_cuts_pruning==0)
        (verbose > 0) && println("Prune cuts ...")
        remove_redundant_cuts!(V)
        for i in 1:length(V)-1
            V[i] = exact_prune_cuts(model, param, V[i])
        end        
        problems = hotstart_SDDP(model, param, V)
    end
end

""" Remove redundant cuts in Polyhedral Value function `V`"""
function remove_cuts(V::PolyhedralFunction)
    Vf = hcat(V.lambdas, V.betas)
    Vf = unique(Vf, 1)
    return PolyhedralFunction(Vf[:, end], Vf[:, 1:end-1], size(Vf)[1])
end


""" Remove redundant cuts in a vector of Polyhedral Functions `Vts`."""
function remove_redundant_cuts!(Vts::Vector{PolyhedralFunction})
    n_functions = length(Vts)-1
    for i in 1:n_functions
        Vts[i] = remove_cuts(Vts[i])
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
