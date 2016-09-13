#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  utility functions
#
#############################################################################


"""
Dump Polyhedral functions in a text file.

# Arguments
* `dump::String`:
    Name of output filt
* `Vts::Vector{PolyhedralFunction}`:
    Vector of polyhedral functions to save
"""
function dump_polyhedral_functions(dump::AbstractString, Vts::Vector{PolyhedralFunction})
    outfile = open(dump, "w")

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


"""
Extract a vector stored in a 3D Array

# Arguments
* `input_array::Array{Float64, 3}`:
    array storing the values of vectors
* `nx::Int64`:
    Position of vector in first dimension
* `ny::Int64`:
    Position of vector in second dimension

# Return
`Vector{Float64}`
"""
function extract_vector_from_3Dmatrix(input_array::Array{Float64, 3},
                                      nx::Int64,
                                      ny::Int64)
    info("extract_vector_from_3Dmatrix is now deprecated. Use collect instead.")
    state_dimension = size(input_array)[3]
    return reshape(input_array[nx, ny, :], state_dimension)
end


"""
Generate a random state.

# Arguments
* `model::SPModel`:

# Return
`Vector{Float64}`, shape=`(model.dimStates,)`

"""
function get_random_state(model)
    return [model.xlim[i][1] + rand()*(model.xlim[i][2] - model.xlim[i][1]) for i in 1:model.dimStates]
end

"""
Estimate upperbound during SDDP iterations.

# Arguments
* `model::SPModel`:
* `params::SDDPparameters`:
* `Vector{PolyhedralFunction}`:
    Polyhedral functions where cuts will be removed
* `iteration_count::Int64`:
    current iteration number
* `upperbound_scenarios`
* `verbose::Int64`

# Return
* `upb::Float64`:
    estimation of upper bound
"""
#TODO à reprendre
function in_iteration_upb_estimation(model::SPModel, 
                    param::SDDPparameters,
                    iteration_count::Int64,
                    verbose::Int64,
                    upperbound_scenarios,
                    current_upb,
                    problems)
        upb = current_upb
        # If specified, compute upper-bound:
        if (param.compute_ub > 0) && (iteration_count%param.compute_ub==0)
            (verbose > 0) && println("Compute upper-bound with ",
                                      param.monteCarloSize, " scenarios...")
            # estimate upper-bound with Monte-Carlo estimation:
            upb, costs = estimate_upper_bound(model, param, upperbound_scenarios, problems)
        end
        return upb
end

"""
Estimate upper bound with Monte Carlo.

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `V::Array{PolyhedralFunction}`:
    the current estimation of Bellman's functions
* `problems::Array{JuMP.Model}`:
    Linear model used to approximate each value function
* `n_simulation::Float64`:
    Number of scenarios to use to compute Monte-Carlo estimation

# Return
* `upb::Float64`:
    estimation of upper bound
* `costs::Vector{Float64}`:
    Costs along different trajectories
"""
function estimate_upper_bound(model::SPModel, param::SDDPparameters,
                                V::Vector{PolyhedralFunction},
                                problem::Vector{JuMP.Model},
                                n_simulation=1000::Int)
    aleas = simulate_scenarios(model.noises, n_simulation)
    return estimate_upper_bound(model, param, aleas, problem)
end
function estimate_upper_bound(model::SPModel, param::SDDPparameters,
                                aleas::Array{Float64, 3},
                                problem::Vector{JuMP.Model})
    costs = forward_simulations(model, param, problem, aleas)[1]
    return upper_bound(costs), costs
end


"""
Estimate the upper bound with a distribution of costs

# Description
Given a probability p, we have a confidence interval:
[mu - alpha sigma/sqrt(n), mu + alpha sigma/sqrt(n)]
where alpha depends upon p.

Upper bound is the max of this interval.

# Arguments
* `cost::Vector{Float64}`:
    Costs values
* `probability::Float`:
    Probability to be inside the confidence interval

# Return
estimated-upper bound as `Float`
"""
function upper_bound(cost::Vector{Float64}, probability=.975)
    tol = sqrt(2) * erfinv(2*probability - 1)
    return mean(cost) + tol*std(cost)/sqrt(length(cost))
end




