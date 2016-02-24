#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Stochastic dynamic programming algorithm
#
#############################################################################

using ProgressMeter

"""
Convert the state and control tuples (stored as arrays) of the problem into integers and vice versa

"""

function index_from_variable( variable::Array,
                    lower_bounds::Array,
                    variable_sizes::Array,
                    variable_steps::Array,
                    model::SPModel,
                    param::SDPparameters)
    index = 0;
    j = 1;

    for i = 1:length(variable)
        index = index + j * round( Int, ( variable[i] - lower_bounds[i] ) / variable_steps[i] )
        j = j * variable_sizes[i]
    end

    return index
end

function variable_from_index( variable_ind::Int,
                    lower_bounds::Array,
                    variable_sizes::Array,
                    variable_steps::Array,
                    model::SPModel,
                    param::SDPparameters)
    dim = length(lower_bounds)
    index = variable_ind
    variable = zeros(dim)

    if (variable_ind>-1)
        for i=1:dim
            variable[i] = lower_bounds[i] + variable_steps[i] * (index % variable_sizes[i])
            index = index -index % variable_sizes[i]
            index = index / variable_sizes[i]
        end
    end

    return variable
end

"""
Value iteration algorithm to compute optimal policy in the Decision Hazard (DH) case and value functions in the HD case

"""

function sdp_optimize_DH(model::SPModel,
                  param::SDPparameters,
                  display=true::Bool)

    V = zeros(Float64, (model.stageNumber+1) * param.totalStateSpaceSize)
    Pi = zeros(Int64, (model.stageNumber+1) * param.totalStateSpaceSize)
    x=zeros(Float64, model.dimStates)
    x1=zeros(Float64, model.dimStates)
    TF = model.stageNumber
    law = model.noises

    u_lower_bounds = [ i for (i , j) in model.ulim]
    x_lower_bounds = [ i for (i , j) in model.xlim]

    count_iteration = 1
    #println("Iteration number : ", count_iteration)
    for indx = 0:(param.totalStateSpaceSize-1)
        x = variable_from_index(indx,
                        x_lower_bounds,
                        param.stateVariablesSizes,
                        param.stateSteps,
                        model,
                        param)
        V[TF+1 + (TF+1) * indx] = model.finalCostFunction(x)
    end

    println("Starting sdp decision hazard policy computation")
    p = Progress(TF*param.totalStateSpaceSize, 1)

    for t = 1:TF
        count_iteration = count_iteration + 1
        #println("Iteration number : ", count_iteration)

        for indx = 0:(param.totalStateSpaceSize-1)
            next!(p)

            v = Inf
            v1 = 0
            indu1 = -1
            Lv = 0
            x = variable_from_index(indx, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)

            for indu = 0:(param.totalControlSpaceSize-1)

                v1 = 0
                count = 0

                for w = 1:param.monteCarloSize

                    u = variable_from_index(indu, u_lower_bounds, param.controlVariablesSizes, param.controlSteps, model, param)
                    wsample = sampling( law, t)
                    x1 = model.dynamics(t, x, u, wsample)

                    if model.constraints(t, x, x1, u, wsample)

                        count = count + 1
                        indx1 = index_from_variable(x1, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)
                        Lv = model.costFunctions(t, x, u, wsample)
                        v1 = v1 + Lv + V[t + 1 + (TF+1) * indx1]

                    end
                end

                if (count>0)

                    v1 = v1 / count

                    if (v1 < v)

                        v = v1
                        indu1 = indu

                    end
                end
            end

            V[t + (TF+1) * indx] = v
            Pi[t + (TF+1) * indx] = indu1

        end
    end

    return V, Pi
end


function sdp_optimize_HD(model::SPModel,
                  param::SDPparameters,
                  display=true::Bool)

    V = zeros(Float64, (model.stageNumber+1) * param.totalStateSpaceSize)
    x=zeros(Float64, model.dimStates)
    x1=zeros(Float64, model.dimStates)
    TF = model.stageNumber
    law = model.noises

    u_lower_bounds = [ i for (i , j) in model.ulim]
    x_lower_bounds = [ i for (i , j) in model.xlim]

    count_iteration = 1
    #println("Iteration number : ", count_iteration)
    for indx = 0:(param.totalStateSpaceSize-1)
        x = variable_from_index(indx, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)

        V[TF+1 + (TF+1) * indx] = model.finalCostFunction(x)
    end

    println("Starting sdp hazard decision policy computation")
    p = Progress(TF*param.totalStateSpaceSize, 1)

    for t = 1:TF
        count_iteration = count_iteration + 1
        #println("Iteration number : ", count_iteration)


        for indx = 0:(param.totalStateSpaceSize-1)
            next!(p)

            v = 0
            indu1 = -1
            Lv = 0
            x = variable_from_index(indx, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)

            count = 0

            for w = 1:param.monteCarloSize

                admissible_u_w_count = 0
                v_x_w = 0
                v_x_w1 = Inf

                for indu = 0:(param.totalControlSpaceSize-1)

                    u = variable_from_index(indu, u_lower_bounds, param.controlVariablesSizes, param.controlSteps, model, param)
                    wsample = sampling( law, t)
                    x1 = model.dynamics(t, x, u, wsample)

                    if model.constraints(t, x, x1, u, wsample)

                        if (admissible_u_w_count == 0)
                            admissible_u_w_count = 1
                        end

                        count = count + admissible_u_w_count
                        indx1 = index_from_variable(x1, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)
                        Lv = model.costFunctions(t, x, u, wsample)
                        v_x_w1 = Lv + V[t + 1 + (TF+1) * indx1]

                        if (v_x_w1 < v_x_w)
                            v_x_w = v_x_w1
                        end
                    end
                end

                v = v + v_x_w
            end

            if (count>0)
                v = v / count
            end

            V[t + (TF+1) * indx] = v

        end
    end

    return V
end

"""
Simulation of optimal control given an initial state and an alea scenario

"""


function sdp_forward_simulation_DH(model::SPModel,
                  param::SDPparameters,
                  scenario::Array,
                  X0::Array,
                  value::Array,
                  policy::Array,
                  display=true::Bool)

    TF = model.stageNumber


    u_lower_bounds = [ i for (i , j) in model.ulim]
    x_lower_bounds = [ i for (i , j) in model.xlim]
    U = zeros(Int64, TF)
    X = zeros(Int64, TF+1)
    X[1] = index_from_variable(X0, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)

    J = value[1 + (TF+1) * X[1]]

    for t = 1:TF

        indx = X[t]
        x = variable_from_index(indx, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)
        indu = policy[t + (TF+1) * indx]
        u = variable_from_index(indu, u_lower_bounds, param.controlVariablesSizes, param.controlSteps, model, param)

        x1 = model.dynamics(t, x, u, scenario[t])
        println(x1)
        X[t+1] = index_from_variable(x1, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)
        println(X[t+1])
        U[t] = indu

    end

    controls = Inf*ones(1, TF, model.dimControls)
    states = Inf*ones(1, TF + 1, model.dimStates)

    for i in 1:length(U)
        Uloc = variable_from_index(U[i], u_lower_bounds, param.controlVariablesSizes, param.controlSteps, model, param)
        println(Uloc)
        for j in 1:length(Uloc)
            controls[1, i, j] = Uloc[j]
        end
    end


    for i in 1:length(X)
        Xloc = variable_from_index(X[i], x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)
        for j in 1:length(Xloc)
            states[1, i, j] = Xloc[j]
        end
    end

    return J, states, controls
end



function sdp_forward_simulation_HD(model::SPModel,
                  param::SDPparameters,
                  scenario::Array,
                  X0::Array,
                  value::Array,
                  display=true::Bool)

    TF = model.stageNumber
    u_lower_bounds = [ i for (i , j) in model.ulim]
    x_lower_bounds = [ i for (i , j) in model.xlim]

    U = zeros(Int64, TF)
    X = zeros(Int64, TF+1)
    X[1] = index_from_variable(X0, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)
    J = 0

    for t = 1:TF

        indx = X[t]
        x = variable_from_index(indx, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)
        xRef = x
        induRef = 0
        uRef = variable_from_index(induRef, u_lower_bounds, param.controlVariablesSizes, param.controlSteps, model, param)
        indxRef = 0
        LvRef = Inf

        for indu = 0:(param.totalControlSpaceSize-1)

            u = variable_from_index(indu, u_lower_bounds, param.controlVariablesSizes, param.controlSteps, model, param)
            x1 = model.dynamics(t, x, u, scenario[t])
            indx1 = index_from_variable(x1, x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)
            if model.constraints(t, x, x1, u, scenario[t])
                Lv = model.costFunctions(t, x, u, scenario[t]) + value[t + 1 + (TF+1) * indx1]
                if (Lv < LvRef)
                    induRef = indu
                    indxRef = indx1
                    LvRef = Lv
                end
            end

        end

        U[t] = induRef
        uRef = variable_from_index(induRef, u_lower_bounds, param.controlVariablesSizes, param.controlSteps, model, param)
        X[t+1] = indxRef
        J = J + model.costFunctions(t, x, uRef, scenario[t])

    end

    J = J + model.finalCostFunction(variable_from_index(X[TF + 1], x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param))

    controls = Inf*ones(1, TF, model.dimControls)
    states = Inf*ones(1, TF + 1, model.dimStates)

    for i in 1:length(U)
        Uloc = variable_from_index(U[i], u_lower_bounds, param.controlVariablesSizes, param.controlSteps, model, param)
        for j in 1:length(Uloc)
            controls[1, i, j] = Uloc[j]
        end
    end


    for i in 1:length(X)
        Xloc = variable_from_index(X[i], x_lower_bounds, param.stateVariablesSizes, param.stateSteps, model, param)
        for j in 1:length(Xloc)
            states[1, i, j] = Xloc[j]
        end
    end

    return J, states, controls
end