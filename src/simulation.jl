#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  SDDP stopping criterion
#############################################################################


"""
Estimate upperbound during SDDP iterations.

# Arguments
* `model::SPModel`:
* `params::SDDPparameters`:
* `iteration_count::Int64`:
    current iteration number
* `upperbound_scenarios`
* `verbosity::Int64`
* `current_upb::Tuple{Float64}`
    Current upper-bound
* `problem::Vector{JuMP.Model}`
    Stages' models

# Return
* `upb, σ, tol`:
    estimation of upper bound with confidence level
"""
function in_iteration_upb_estimation(model::SPModel,
                                     param::SDDPparameters,
                                     iteration_count::Int64,
                                     verbosity::Int64,
                                     upperbound_scenarios,
                                     current_upb,
                                     problems)
    upb, σ, tol = current_upb
    # If specified, compute upper-bound:
    if (param.compute_ub > 0) && (iteration_count%param.compute_ub==0)
        (verbosity > 0) && println("Compute upper-bound with ",
                                    param.in_iter_mc, " scenarios...")
        # estimate upper-bound with Monte-Carlo estimation:
        upb, σ, tol = estimate_upper_bound(model, param, upperbound_scenarios, problems)
    end
    return [upb, σ, tol]
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
* `upb, σ, tol`:
    estimation of upper bound
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
    # discard unvalid values:
    costs = costs[isfinite.(costs)]
    μ = mean(costs)
    σ = std(costs)
    tol = upper_bound_confidence(costs, param.confidence_level)
    return μ, σ, tol
end


"""Run `nsimu` simulations of SDDP."""
function simulate(sddp::SDDPInterface, nsimu::Int)
    scenarios = simulate_scenarios(sddp.spmodel.noises, nsimu)
    forward_simulations(sddp.spmodel, sddp.params, sddp.solverinterface, scenarios)[1:3]
end

simulate(sddp::SDDPInterface, scen::Array{Float64, 3}) = forward_simulations(sddp.spmodel, sddp.params, sddp.solverinterface, scen)[1:3]

function upperbound(sddp::SDDPInterface, scenarios::Array)
    costs = forward_simulations(sddp.spmodel, sddp.params, sddp.solverinterface, scenarios)[1]
    # discard unvalid values:
    costs = costs[isfinite.(costs)]
    μ = mean(costs)
    σ = std(costs)
    tol = upper_bound_confidence(costs, sddp.params.confidence_level)
    return [μ, σ, tol]
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
function upper_bound_confidence(cost::Vector{Float64}, probability=.975)
    tol = sqrt(2) * erfinv(2*probability - 1)
    return tol*std(cost)/sqrt(length(cost))
end

