#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  the actual optimization function
#
#############################################################################


include("forwardBackwardIterations.jl")
include("utility.jl")
include("simulate.jl")
include("objects.jl")

"""Build a collection of cuts initialize at 0"""
function get_null_value_functions_array(model::SPModel)

    V = Vector{SDDP.PolyhedralFunction}(model.stageNumber)
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

"""
function build_terminal_cost(problem)
    alpha = getVar(problem, :alpha)
    @addConstraint(problem, alpha >= 0)
end



"""
Initialize each linear problem used to approximate value functions

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

      @defVar(m,  model.xlim[1] <= x[1:nx] <= model.xlim[2])
      @defVar(m,  model.ulim[1] <= u[1:nu] <=  model.ulim[2])
      @defVar(m,  model.xlim[1] <= xf[1:nx]<= model.xlim[2])
      @defVar(m, alpha)

      @defVar(m, w[1:nw] == 0)
      m.ext[:cons] = @addConstraint(m, state_constraint, x .== 0)

      @addConstraint(m, xf .== model.dynamics(x, u, w))

      @setObjective(m, Min, model.costFunctions(t, x, u, w) + alpha)

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
function initialize_value_functions( model::LinearDynamicLinearCostSPmodel,
                                     param::SDDPparameters,
                        )

    n = param.forwardPassNumber

    solverProblems = build_models(model, param)
    solverProblems_null = build_models(model, param)

    V_null = get_null_value_functions_array(model)
    V = Array{PolyhedralFunction}(model.stageNumber)

    # Build scenarios according to distribution laws:
    aleas = simulate_scenarios(model.noises,
                               (model.stageNumber,
                                n,
                                model.dimNoises))


    V[end] = SDDP.PolyhedralFunction(zeros(1), zeros(1, 1), 1)


    stockTrajectories = forward_simulations(model,
                        param,
                        V_null,
                        solverProblems_null,
                        n,
                        aleas,
                        false, true, false)[2]

    build_terminal_cost(solverProblems[end-1])

    backward_pass(model,
                  param,
                  V,
                  solverProblems,
                  stockTrajectories,
                  model.noises,
                  true)

    return V, solverProblems
end



"""
Make a forward pass of the algorithm

Simulate a scenario of noise and compute an optimal trajectory on this
scenario according to the current value functions.

Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- n_iterations (Int) - Default is 20
    Maximum number of iterations to run

- display (Bool) - Default is false
    If specified, display progression in terminal


Returns :
- V (Array{PolyhedralFunction})
    the collection of approximation of the bellman functions

- problems (Array{JuMP.Model})
    the collection of linear problems used to approximate
    each value function

"""
function optimize(model::SPModel,
                  param::SDDPparameters,
                  n_iterations=20,
                  display=true)

    # Initialize value functions:
    V, problems = initialize_value_functions(model, param)

    if display
      println("Initialize cuts")
    end

    # Build given number of scenarios according to distribution
    # law specified in model.noises:
    aleas = simulate_scenarios(model.noises ,
                                (model.stageNumber,
                                 param.forwardPassNumber,
                                 model.dimNoises))

    stopping_test::Bool = false
    iteration_count::Int64 = 0

    n = param.forwardPassNumber

    for i = 1:n_iterations
        stockTrajectories = forward_simulations(model,
                            param,
                            V,
                            problems,
                            n,
                            aleas)[2]

        backward_pass(model,
                      param,
                      V,
                      problems,
                      stockTrajectories,
                      model.noises)

        iteration_count+=1;

        if display
          println("Pass number ", i)
        end
    end

    return V, problems
end
