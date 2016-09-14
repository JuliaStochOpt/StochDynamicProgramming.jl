#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Implement the SDDP solver and initializers:
#  - functions to initialize value functions
#  - functions to build terminal cost
#############################################################################


"""
Solve SDDP algorithm and return estimation of bellman functions.

# Description
Alternate forward and backward phase till the stopping criterion is
fulfilled.

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `verbose::Int64`:
    Default is `0`
    If non null, display progression in terminal every
    `n` iterations, where `n` is the number specified by display.

# Returns
* `V::Array{PolyhedralFunction}`:
    the collection of approximation of the bellman functions
* `problems::Array{JuMP.Model}`:
    the collection of linear problems used to approximate
    each value function
* `sddp_stats::SDDPStat`:

"""
function solve_SDDP(model::SPModel, param::SDDPparameters, verbose=0::Int64)
    check_SDDPparameters(model,param,verbose)
    # initialize value functions:
    V, problems = initialize_value_functions(model, param)
    (verbose > 0) && println("Initial value function loaded into memory.")
    # Run SDDP:
    sddp_stats = run_SDDP!(model, param, V, problems, verbose)
    return V, problems, sddp_stats
end

"""
Solve SDDP algorithm with hotstart and return estimation of bellman functions.

# Description
Alternate forward and backward phase till the stopping criterion is
fulfilled.

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `V::Vector{PolyhedralFunction}`:
    current lower approximation of Bellman functions
* `verbose::Int64`:
    Default is `0`
    If non null, display progression in terminal every
    `n` iterations, where `n` is the number specified by display.

# Returns
* `V::Array{PolyhedralFunction}`:
    the collection of approximation of the bellman functions
* `problems::Array{JuMP.Model}`:
    the collection of linear problems used to approximate
    each value function
* `sddp_stats::SDDPStat`:
"""
function solve_SDDP(model::SPModel, param::SDDPparameters, V::Vector{PolyhedralFunction}, verbose=0::Int64)
    check_SDDPparameters(model,param,verbose)
    # First step: process value functions if hotstart is called
    problems = hotstart_SDDP(model, param, V)
    sddp_stats = run_SDDP!(model, param, V, problems, verbose)
    return V, problems, sddp_stats
end


"""Run SDDP iterations.

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `V::Vector{PolyhedralFunction}`:
    Polyhedral lower approximation of Bellman functions
* `problems::Vector{JuMP.Model}`:
* `verbose::Int64`:
    Default is `0`
    If non null, display progression in terminal every
    `n` iterations, where `n` is the number specified by display.

# Returns
* `stats:SDDPStats`:
    contains statistics of the current algorithm
"""
function run_SDDP!(model::SPModel,
                    param::SDDPparameters,
                    V::Vector{PolyhedralFunction},
                    problems::Vector{JuMP.Model},
                    verbose=0::Int64)

    #Initialization of the counter
    stats = SDDPStat(0, [], [], [], 0)

    (verbose > 0) && println("Initialize cuts")

    # If computation of upper-bound is needed, a set of scenarios is built
    # to keep always the same realization for upper bound estimation:
    #if param.compute_ub > 0 #TODO 
    upperbound_scenarios = simulate_scenarios(model.noises, param.in_iter_mc)
    #end

    upb = Inf
    costs = nothing
    stopping_test::Bool = false
    

    # Launch execution of forward and backward passes:
    while (~stopping_test)
        # Time execution of current pass:
        tic()

        ####################
        # Forward pass : compute stockTrajectories
        costs, stockTrajectories, callsolver_forward = forward_pass!(model,param,V,problems)

        ####################
        # Backward pass : update polyhedral approximation of Bellman functions
        callsolver_backward = backward_pass!(model,param,V,problems,stockTrajectories,model.noises)
        
        ####################
        # cut pruning
        prune_cuts!(model,param,V,stats.niterations,verbose)
 
        ####################
        # In iteration upper bound estimation
        upb = in_iteration_upb_estimation(model,param,stats.niterations,verbose,
                                            upperbound_scenarios,upb,problems)        

        ####################
        # Update stats 
        lwb = get_bellman_value(model, param, 1, V[1], model.initialState)  
        updateSDDPStat!(stats,callsolver_forward + callsolver_backward,lwb,upb,toq())
        
        print_current_stats(stats,verbose)
        
        ####################
        # Stopping test
        stopping_test = test_stopping_criterion(param,stats)
    end

    ##########
    # Estimate final upper bound with param.monteCarloSize simulations:
    sddp_finish(model, param,V,problems,stats,verbose)
    return stats
end

function sddp_finish(model::SPModel, param::SDDPparameters,V,problems,stats::SDDPStat,verbose::Int64)
    if (verbose>0) && (param.compute_ub >= 0)
        lwb = get_bellman_value(model, param, 1, V[1], model.initialState)
        
        if param.compute_ub == 0
            println("Estimate upper-bound with Monte-Carlo ...")
            upb, costs = estimate_upper_bound(model, param, V, problems, param.monteCarloSize)
        else
            upb = stats.upperbounds[end]
        end

        println("Estimation of upper-bound: ", round(upb,4),
                "\tExact lower bound: ", round(lwb,4),
                "\t Gap <  ", round(100*(upb-lwb)/lwb, 2) , "\%  with prob. > 97.5 \%")
        println("Estimation of cost of the solution (fiability 95\%):",
                 round(mean(costs),4), " +/- ", round(1.96*std(costs)/sqrt(length(costs)),4))
    end
end



"""
Build a cut approximating terminal cost with null function

# Arguments
* `problem::JuMP.Model`:
    Cut approximating the terminal cost
* `shape`:
    If PolyhedralFunction is given, build terminal cost with it
    Else, terminal cost is null
"""
function build_terminal_cost!(model::SPModel, problem::JuMP.Model, Vt::PolyhedralFunction)
    # if shape is PolyhedralFunction, build terminal cost with it:
    alpha = getvariable(problem, :alpha)
    xf = getvariable(problem, :xf)
    t = model.stageNumber -1
    if isa(Vt, PolyhedralFunction)
        for i in 1:Vt.numCuts
            lambda = vec(Vt.lambdas[i, :])
            @constraint(problem, Vt.betas[i] + dot(lambda, xf) <= alpha)
        end
    else
        @constraint(problem, alpha >= 0)
    end
end


"""
Initialize each linear problem used to approximate value  functions

# Description
This function define the variables and the constraints of each
linear problem.

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP

# Return
* `Array::JuMP.Model`:
"""
function build_models(model::SPModel, param::SDDPparameters)
    return JuMP.Model[build_model(model, param, t) for t=1:model.stageNumber-1]
end

function build_model(model, param, t)
    m = Model(solver=param.SOLVER)

    nx = model.dimStates
    nu = model.dimControls
    nw = model.dimNoises

    # define variables in JuMP:
    @variable(m,  model.xlim[i][1] <= x[i=1:nx] <= model.xlim[i][2])
    @variable(m,  model.xlim[i][1] <= xf[i=1:nx]<= model.xlim[i][2])
    @variable(m,  model.ulim[i][1] <= u[i=1:nu] <=  model.ulim[i][2])
    @variable(m, alpha)

    @variable(m, w[1:nw] == 0)
    m.ext[:cons] = @constraint(m, state_constraint, x .== 0)

    @constraint(m, xf .== model.dynamics(t, x, u, w))

    # Add equality and inequality constraints:
    if model.equalityConstraints != nothing
        @constraint(m, model.equalityConstraints(t, x, u, w) .== 0)
    end
    if model.inequalityConstraints != nothing
        @constraint(m, model.inequalityConstraints(t, x, u, w) .<= 0)
    end

    # Define objective function (could be linear or piecewise linear)
    if isa(model.costFunctions, Function)
        @objective(m, Min, model.costFunctions(t, x, u, w) + alpha)
    elseif isa(model.costFunctions, Vector{Function})
        @variable(m, cost)

        for i in 1:length(model.costFunctions)
            @constraint(m, cost >= model.costFunctions[i](t, x, u, w))
        end
        @objective(m, Min, cost + alpha)
    end

    # Add binary variable if problem is a SMIP:
    if model.IS_SMIP
        m.colCat[2*nx+1:2*nx+nu] = model.controlCat
    end

    return m
end


"""
Initialize value functions along a given trajectory

# Description
This function add the fist cut to each PolyhedralFunction stored in a Array

# Arguments
* `model::SPModel`:
* `param::SDDPparameters`:

# Return
* `V::Array{PolyhedralFunction}`:
    Return T PolyhedralFunction, where T is the number of stages
    specified in model.
* `problems::Array{JuMP.Model}`:
    the initialization of linear problems used to approximate
    each value function
"""
function initialize_value_functions(model::SPModel,
                                    param::SDDPparameters)

    solverProblems = build_models(model, param)
    V = PolyhedralFunction[
                PolyhedralFunction(model.dimStates) for i in 1:model.stageNumber]

    # Build scenarios according to distribution laws:
    aleas = simulate_scenarios(model.noises, param.forwardPassNumber)

    # Add final costs to solverProblems:
    if isa(model.finalCost, PolyhedralFunction)
        V[end] = model.finalCost
        build_terminal_cost!(model, solverProblems[end], V[end])
    elseif isa(model.finalCost, Function)
        model.finalCost(model, solverProblems[end])
    end

    stockTrajectories = zeros(model.stageNumber, param.forwardPassNumber, model.dimStates)
    for i in 1:model.stageNumber, j in 1:param.forwardPassNumber
        stockTrajectories[i, j, :] = get_random_state(model)
    end

    callsolver = backward_pass!(model, param, V, solverProblems,
                                stockTrajectories, model.noises)

    return V, solverProblems
end


"""
Initialize JuMP.Model vector with a previously computed PolyhedralFunction
vector.

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP
* `V::Vector{PolyhedralFunction}`:
    Estimation of bellman functions as Polyhedral functions

# Return
* `Vector{JuMP.Model}`
"""
function hotstart_SDDP(model::SPModel, param::SDDPparameters, V::Vector{PolyhedralFunction})

    solverProblems = build_models(model, param)

    # Add corresponding cuts to each problem:
    for t in 1:model.stageNumber-2
        add_cuts_to_model!(model, t, solverProblems[t], V[t+1])
    end

    # Take care of final cost:
    if isa(model.finalCost, PolyhedralFunction)
        add_cuts_to_model!(model, model.stageNumber-1, solverProblems[end], V[end])
    else
        model.finalCost(model, solverProblems[end])
    end
    return solverProblems
end


"""
Compute value of Bellman function at point xt. Return V_t(xt)

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP
* `t::Int64`:
    Time t where to solve bellman value
* `Vt::Polyhedral function`:
    Estimation of bellman function as Polyhedral function
* `xt::Vector{Float64}`:
    Point where to compute Bellman value.

# Return
Bellman value (Float64)
"""
function get_bellman_value(model::SPModel, param::SDDPparameters,
                           t::Int64, Vt::PolyhedralFunction, xt::Vector{Float64})

    m = Model(solver=param.SOLVER)
    @variable(m, alpha)

    for i in 1:Vt.numCuts
        lambda = vec(Vt.lambdas[i, :])
        @constraint(m, Vt.betas[i] + dot(lambda, xt) <= alpha)
    end

    @objective(m, Min, alpha)
    solve(m)
    return getvalue(alpha)
end


"""
Compute lower-bound of the problem at initial time.

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP
* `V::Vector{Polyhedral function}`:
    Estimation of bellman function as Polyhedral function

# Return
current lower bound of the problem (Float64)
"""
function get_lower_bound(model::SPModel, param::SDDPparameters,
                            V::Vector{PolyhedralFunction})
    return get_bellman_value(model, param, 1, V[1], model.initialState)
end


"""
Compute optimal control at point xt and time t.

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP
* `lpproblem::Vector{JuMP.Model}`:
    Linear problems used to approximate the value functions
* `t::Int64`:
    Time
* `xt::Vector{Float64}`:
    Position where to compute optimal control
* `xi::Vector{Float64}`:
    Alea at time t

# Return
    `Vector{Float64}`: optimal control at time t
"""
function get_control(model::SPModel, param::SDDPparameters, lpproblem::Vector{JuMP.Model},
                     t::Int, xt::Vector{Float64}, xi::Vector{Float64})
    return solve_one_step_one_alea(model, param, lpproblem[t], t, xt, xi)[2].optimal_control
end


"""
Add several cuts to JuMP.Model from a PolyhedralFunction

# Arguments
* `model::SPModel`:
    Store the problem definition
* `t::Int`:
    Time index
* `problem::JuMP.Model`:
    Linear problem used to approximate the value functions
* `V::PolyhedralFunction`:
    Cuts are stored in V
"""
function add_cuts_to_model!(model::SPModel, t::Int64, problem::JuMP.Model, V::PolyhedralFunction)
    alpha = getvariable(problem, :alpha)
    xf = getvariable(problem, :xf)

    for i in 1:V.numCuts
        lambda = vec(V.lambdas[i, :])
        @constraint(problem, V.betas[i] + dot(lambda, xf) <= alpha)
    end
end
