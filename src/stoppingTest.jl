"""
Test if the stopping criteria is fulfilled.

Return true if |upper_bound - lower_bound|/lower_bound < epsilon
or iteration_count > maxItNumber

# Arguments
*`SDDPparameters`:
    stopping test type defined in SDDPparameters
* `stats::SDDPStat`:
    statistics of the current algorithm

# Return
`Bool`
"""
function test_stopping_criterion(param::SDDPparameters, stats::SDDPStat)
    lb = stats.lower_bounds[end]
    ub = stats.upper_bounds[end]
    check_gap = (abs((ub-lb)/lb) < param.gap)
    check_iter = stats.niterations > param.maxItNumber
    return check_gap || check_iter
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
                                      param.in_iter_mc, " scenarios...")
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
