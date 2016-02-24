#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  the actual optimization function
#
#############################################################################


"""
Solve SDDP algorithm and return estimation of bellman functions.

Alternate forward and backward phase till the stopping criterion is
fulfilled.


Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- display (Bool) - Default is false
    If specified, display progression in terminal


Returns :
- V (Array{PolyhedralFunction})
    the collection of approximation of the bellman functions

- problems (Array{JuMP.Model})
    the collection of linear problems used to approximate
    each value function

"""
function solve_SDDP(model::SPModel,
                    param::SDDPparameters,
                    display=0::Int64,
                    returnValueFunctions=true::Bool)

    # Initialize value functions:
    V, problems = initialize_value_functions(model, param)
    # Evaluation of initial cost:
    V0::Float64 = 0

    if display > 0
      println("Initialize cuts")
    end


    stopping_test::Bool = false
    iteration_count::Int64 = 0

    while (iteration_count < param.maxItNumber) & (~stopping_test)
        # Time execution of current pass:
        tic()

        # Build given number of scenarios according to distribution
        # law specified in model.noises:
        aleas = simulate_scenarios(model.noises ,
                                    (model.stageNumber,
                                     param.forwardPassNumber,
                                     model.dimNoises))

        # Forward pass
        costs, stockTrajectories, _ = forward_simulations(model,
                            param,
                            V,
                            problems,
                            aleas)

        # Backward pass
        backward_pass!(model,
                      param,
                      V,
                      problems,
                      stockTrajectories,
                      model.noises,
                      false,
                      returnValueFunctions)

        iteration_count += 1
        upb = upper_bound(costs)

        V0 = get_bellman_value(model, param, 1, V[1], model.initialState)

        time = toq()

        if (display > 0) && (iteration_count%display==0)
            println("Pass number ", iteration_count,
                    "\tEstimation of upper-bound: ", upb,
                    "\tLower-bound: ", V0,
                    "\tTime: ", time)
        end

    end

    if (display>0)
        println("Estimate upper-bound with Monte-Carlo ...")
        upb, costs = estimate_upper_bound(model, param, V, problems)
        println("Estimation of upper-bound: ", upb,
                "\tExact lower bound: ", V0,
                "\t Gap (\%) <  ", (V0-upb)/V0 , " with prob. > 97.5 \%")
        println("Estimation of cost of the solution (fiability 95\%):",
                 mean(costs), " +/- ", 1.96*std(costs)/sqrt(length(costs)))
    end

    return V, problems
end



"""
Estimate upper bound with Monte Carlo.

Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- V (Array{PolyhedralFunction})
    the current estimation of Bellman's functions

- problems (Array{JuMP.Model})
    Linear model used to approximate each value function

- n_simulation (Float64)
    Number of scenarios to use to compute Monte-Carlo estimation


Return:
Float64 (estimation of the upper bound)

"""
function estimate_upper_bound(model, param, V, problems, n_simulation=1000)

    n_fpn = param.forwardPassNumber
    param.forwardPassNumber = n_simulation

    aleas = simulate_scenarios(model.noises ,
                                    (model.stageNumber,
                                     param.forwardPassNumber,
                                     model.dimNoises))

    costs, stockTrajectories, _ = forward_simulations(model,
                                                        param,
                                                        V,
                                                        problems,
                                                        aleas)


    param.forwardPassNumber = n_fpn

    return upper_bound(costs), costs
end



"""Build a collection of cuts initialize at 0"""
function get_null_value_functions_array(model::SPModel)

    V = Vector{PolyhedralFunction}(model.stageNumber)
    for t = 1:model.stageNumber
        V[t] = get_null_value_functions()
    end

    return V
end



"""
Build a cut approximating terminal cost with null function


Parameter:
- problem (JuMP.Model)
    Cut approximating the terminal cost

- shape
    If PolyhedralFunction is given, build terminal cost with it
    Else, terminal cost is null

"""
function build_terminal_cost!(model::SPModel, problem::JuMP.Model, Vt)
    alpha = getVar(problem, :alpha)

    # if shape is PolyhedralFunction, build terminal cost with it:
    if isa(Vt, PolyhedralFunction)
        alpha = getVar(problem, :alpha)
        x = getVar(problem, :x)
        u = getVar(problem, :u)
        w = getVar(problem, :w)
        t = model.stageNumber -1
        for i in 1:Vt.numCuts
            lambda = vec(Vt.lambdas[i, :])
            @addConstraint(problem, Vt.betas[i] + dot(lambda, model.dynamics(t, x, u, w)) <= alpha)
        end
    else
        @addConstraint(problem, alpha >= 0)
    end
end



"""
Initialize each linear problem used to approximate value  functions

This function define the variables and the constraints of each
linear problem.


Parameter:
- model (SPModel)
    Parametrization of the problem

- param (SDDPparameters)
    Parameters of SDDP


Return:
- Array{JuMP.Model}

"""
function build_models(model::SPModel, param::SDDPparameters)

    models = Vector{JuMP.Model}(model.stageNumber)


    for t = 1:model.stageNumber
        m = Model(solver=param.solver)

        nx = model.dimStates
        nu = model.dimControls
        nw = model.dimNoises

        @defVar(m,  model.xlim[i][1] <= x[i=1:nx] <= model.xlim[i][2])
        @defVar(m,  model.ulim[i][1] <= u[i=1:nu] <=  model.ulim[i][2])
        @defVar(m,  model.xlim[i][1] <= xf[i=1:nx]<= model.xlim[i][2])
        @defVar(m, alpha)

        @defVar(m, w[1:nw] == 0)
        m.ext[:cons] = @addConstraint(m, state_constraint, x .== 0)

        @addConstraint(m, xf .== model.dynamics(t, x, u, w))

        if typeof(model) == LinearDynamicLinearCostSPmodel
            @setObjective(m, Min, model.costFunctions(t, x, u, w) + alpha)

        elseif typeof(model) == PiecewiseLinearCostSPmodel
            @defVar(m, cost)

            for i in 1:length(model.costFunctions)
                @addConstraint(m, cost >= model.costFunctions[i](t, x, u, w))
            end
            @setObjective(m, Min, cost + alpha)

        else
            error("model must be: LinearDynamicLinearCostSPModel or LinearDynamicLinearCostSPmodel")
        end

        models[t] = m

    end
    return models
end



"""
Initialize value functions along a given trajectory

This function add the fist cut to each PolyhedralFunction stored in a Array


Parameters:
- model (SPModel)

- param (SDDPparameters)

Return:
- V (Array{PolyhedralFunction})
    Return T PolyhedralFunction, where T is the number of stages
    specified in model.

- problems (Array{JuMP.Model})
    the initialization of linear problems used to approximate
    each value function

"""
function initialize_value_functions( model::SPModel,
                                     param::SDDPparameters)

    solverProblems = build_models(model, param)
    solverProblems_null = build_models(model, param)

    V_null = get_null_value_functions_array(model)
    V = Array{PolyhedralFunction}(model.stageNumber)

    # Build scenarios according to distribution laws:
    aleas = simulate_scenarios(model.noises,
                               (model.stageNumber,
                                param.forwardPassNumber,
                                model.dimNoises))


    V[end] = PolyhedralFunction(zeros(1), zeros(1, model.dimStates), 1)


    stockTrajectories = forward_simulations(model,
                        param,
                        V_null,
                        solverProblems_null,
                        aleas,
                        false, true, false)[2]

    build_terminal_cost!(model, solverProblems[end-1], V[end])

    backward_pass!(model,
                  param,
                  V,
                  solverProblems,
                  stockTrajectories,
                  model.noises,
                  true,
                  true)

    return V, solverProblems
end



"""
Compute value of Bellman function at point xt. Return V_t(xt)

Parameters:
- model (SPModel)
    Parametrization of the problem

- param (SDDPparameters)
    Parameters of SDDP

- t (Int64)
    Time t where to solve bellman value

- Vt (Polyhedral function)
    Estimation of bellman function as Polyhedral function

- xt (Vector{Float64})
    Point where to compute Bellman value.


Return:
Bellman value (Float64)

"""
function get_bellman_value(model::SPModel, param::SDDPparameters,
                           t::Int64, Vt::PolyhedralFunction, xt::Vector{Float64})

    m = Model(solver=param.solver)
    @defVar(m, alpha)

    for i in 1:Vt.numCuts
        lambda = vec(Vt.lambdas[i, :])
        @addConstraint(m, Vt.betas[i] + dot(lambda, xt) <= alpha)
    end

    @setObjective(m, Min, alpha)
    solve(m)
    return getValue(alpha)
end
