#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Model and solve the One-Step One Alea problem in different settings
# - used to compute the optimal control (in forward phase / simulation)
# - used to compute the cuts in the Backward phase
#############################################################################

"""
Solve the Bellman equation at time t starting at state x under alea xi
with the current evaluation of Vt+1

# Description
The function solve
min_u current_cost(t,x,u,xi) + current_Bellman_Value_{t+1}(dynamic(t,x,u,xi))
and can return the optimal control and a subgradient of the value of the
problem with respect to the initial state x

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `m::JuMP.Model`:
    The linear problem to solve, in order to approximate the
    current value functions
* `t::int`:
    time step at which the problem is solved
* `xt::Array{Float}`:
    current starting state
* `xi::Array{float}`:
    current noise value
* `relaxation::Bool`: default is false
    If problem is MILP, specify if it is needed to relax integer constraints.
* `init::Bool`:
    If specified, approximate future cost as 0

# Returns
* `solved::Bool`:
    True if the solution is feasible, false otherwise
* `NextStep`:
    Store solution of the problem
* `ts::Float64`:
   Solver's execution time
"""
function solve_one_step_one_alea(model,
                                 param,
                                 m::JuMP.Model,
                                 t::Int64,
                                 xt::Vector{Float64},
                                 xi::Vector{Float64};
                                 relaxation=false::Bool,
                                 init=false::Bool,
                                 verbosity::Int64=0)

    # Get var defined in JuMP.model:
    x = getindex(m, :x)
    u = getindex(m, :u)
    w = getindex(m, :w)
    alpha = getindex(m, :alpha)

    # Update value of w:
    JuMP.fix.(w,xi)

    #update objective
    if isa(model.costFunctions, Function)
        try
            @objective(m, Min, model.costFunctions(t, x, u, xi) + alpha)
        catch
            @objective(m, Min, model.costFunctions(m, t, x, u, xi) + alpha)
        end
    elseif isa(model.costFunctions, Vector{Function})
        cost = getindex(m, :cost)
        for i in 1:length(model.costFunctions)
            @constraint(m, cost >= model.costFunctions[i](t, x, u, xi))
        end
        @objective(m, Min, cost + alpha)
     end

    # Update constraint x == xt

    for i in 1:model.dimStates
        JuMP.set_normalized_rhs(m.ext[:cons][i], xt[i])
    end

    if false
        println("One step one alea problem at time t=",t)
        println("for x =",xt)
        println("and w=",xi)
        print(m)
    end

    if model.IS_SMIP
        solved = relaxation ? solve_relaxed!(m, param,verbosity) : solve_mip!(m, param,verbosity)
    else
        remove_infinite_constraint!(m)
        JuMP.optimize!(m)
        status = JuMP.termination_status(m)
        solved = (status == MOI.OPTIMAL) || (status == MOI.TIME_LIMIT && JuMP.has_values(m))
    end
    # get time taken by the solver:

    solvetime = try JuMP.solve_time(m) catch e 0.0 end
    if solved
        optimalControl = JuMP.value.(u)
        λ = JuMP.has_duals(m) ? Float64[JuMP.dual(m.ext[:cons][i]) for i in 1:model.dimStates] : Array{Float64, 1}()
        # Return object storing results:
        result = NLDSSolution(
                          solved,
                          JuMP.objective_value(m),
                          model.dynamics(t, xt, optimalControl, xi),
                          optimalControl,
                          λ,
                          JuMP.value.(alpha),
                          getcutsmultipliers(m))
    else
        println(m)
        println(status)
        error("Foo")
        # If no solution is found, then return nothing
        #= println(m) =#
        #= error("Fail to solve") =#
        result = NLDSSolution()
    end
    return result, solvetime
end


"""Solve model in Decision-Hazard."""
function solve_dh(model, param, t, xt, m; verbosity::Int64=0)
    xf = getindex(m, :xf)
    u = getindex(m, :u)
    alpha = getindex(m, :alpha)
    for i in 1:model.dimStates
        JuMP.set_normalized_rhs(m.ext[:cons][i], xt[i])
    end

    (verbosity>5) && println("Decision Hazard model")
    (verbosity>5) && print(m)

    remove_infinite_constraint!(m)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    solved = status == MOI.OPTIMAL

    if ~solved
        println(m)
        println(JuMP.value.(u))
        println(JuMP.value.(alpha))
        println(JuMP.value.(xf))
        @warn("dh model not solved at time t=",t)
    end

    solvetime = try JuMP.solve_time(m) catch e 0.0 end

    if solved
        # Computation of subgradient:
        λ = JuMP.has_duals(m) ? Float64[JuMP.dual(m.ext[:cons][i]) for i in 1:model.dimStates] : Array{Float64, 1}()
        result = DHNLDSSolution(solved,
                              JuMP.objective_value(m),
                              JuMP.value.(xf),
                              JuMP.value.(u)[:, 1],
                              λ,
                              JuMP.value.(alpha),
                              getcutsmultipliers(m))
    else
        # If no solution is found, then return nothing
        result = NLDSSolution()
    end

    return result, solvetime
end



# Solve local problem with a quadratic penalization:
function regularize(model, param,
                    regularizer::AbstractRegularization,
                    m::JuMP.Model,
                    t::Int64,
                    xt::Vector{Float64}, xi::Vector{Float64}, xp::Vector{Float64};verbosity=0::Int64)
    # store current objective:
    pobj = m.obj
    xf = getindex(m, :xf)
    qexp = getpenaltyexpr(regularizer, xf, xp)
    # and update model objective:
    @objective(m, :Min, m.obj + qexp)
    res = solve_one_step_one_alea(model, param, m, t, xt, xi,verbosity=verbosity)
    m.obj = pobj

    return res
end


"""Solve relaxed MILP problem."""
function solve_relaxed!(m, param,verbosity::Int64=0)
    JuMP.set_optimizer(m, param.OPTIMIZER)
    remove_infinite_constraint!(m)
    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    return status == MOI.OPTIMAL
end


"""Solve original MILP problem."""
function solve_mip!(m, param,verbosity::Int64=0)
    JuMP.set_optimizer(m, param.MIPOPTIMIZER)
    status = JuMP.termination_status(m)
    return status == MOI.OPTIMAL
end


getcutsmultipliers(m::JuMP.Model)= try _getdual(m)[end-m.ext[:ncuts]+1:end] catch e Array{Float64,1}() end
function _getdual(m::JuMP.Model)
    return JuMP.has_duals(m) ? JuMP.dual.([m.ext[:cons][i] for i in 1:length(m.ext[:cons])]) : Array{Float64,1}()
end
