#  Copyright 2015, Vincent Leclere, Francois Pacaud, Henri Gerard and
#  Tristan Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Stochastic dynamic programming algorithm
#
#############################################################################

using ProgressMeter
using Iterators
using Interpolations

"""
Convert the state and control float tuples (stored as arrays or tuples) of the
problem into integer tuples that can be used as indexes for the value function

Parameters:
- variable (Array)
    the vector variable we want to convert to an index (integer)

- bounds (Array)
    the lower bounds for each component of the variable

- variable_steps (Array)
    discretization step for each component


Returns :
- index (tuple of integers)
    the indexes of the variable

"""
function index_from_variable( variable,
                    bounds::Array,
                    variable_steps::Array)
    index = tuple();

    for i = 1:length(variable)
        index = tuple(index..., floor(Int64, 1 + fld(1e-10+( variable[i] - bounds[i][1] ), variable_steps[i] )))
    end

    return index
end

"""
Convert the state and control float tuples (stored as arrays or tuples) of the
problem into float tuples that can be used as indexes for the interpolated
value function

Parameters:
- variable (Array)
    the vector variable we want to convert to an index (integer)

- bounds (Array)
    the lower bounds for each component of the variable

- variable_steps (Array)
    discretization step for each component


Returns :
- index (tuple of integers)
    the indexes of the variable

"""
function real_index_from_variable( variable,
                    bounds::Array,
                    variable_steps::Array)
    index = tuple();

    for i = 1:length(variable)
        index = tuple(index..., 1 + (( variable[i] - bounds[i][1] )/variable_steps[i] ))
    end

    return index
end

"""
Compute interpolation of the value function at time t

Parameters:
- model (SPmodel)

- v (Array)
    the value function to interpolate

- time (Int)
    time at which we have to interpolate V


Returns :
- Interpolation
    the interpolated value function (working as an array with float indexes)
"""
function value_function_interpolation( model::SPModel,
                                    V::Array,
                                    time::Int)
    columns = Array{Colon}(model.dimStates)
    for i in 1:model.dimStates
        columns[i] = Colon()
    end

    return interpolate(V[columns...,time], BSpline(Linear()), OnGrid())
end

"""
Compute interpolation of the value function at time t

Parameters:
- model (SPmodel)
    the model of the problem

- param (SDPparameters)
    the parameters of the problem


Returns :
- Iterators : product_states and product_controls
    the cartesian product iterators for both states and controls
"""
function generate_grid(model::SPModel, param::SDPparameters)

    tab_states = Array{FloatRange}(model.dimStates)
    tab_controls = Array{FloatRange}(model.dimControls)


    for i = 1:model.dimStates
        tab_states[i] = model.xlim[i][1]:param.stateSteps[i]:model.xlim[i][2]
    end

    for i = 1:model.dimControls
        tab_controls[i] = model.ulim[i][1]:param.controlSteps[i]:model.ulim[i][2]
    end

    product_states = product(tab_states...)

    product_controls = product(tab_controls...)

    return product_states, product_controls

end

"""
Value iteration algorithm to compute optimal value functions in
the Decision Hazard (DH) as well as the Hazard Decision (HD) case

Parameters:
- model (SPmodel)
    the DPSPmodel of our problem

- param (SDPparameters)
    the parameters for the SDP algorithm

- display (Bool)
    the output display or verbosity parameter


Returns :
- value_functions (Array)
    the vector representing the value functions as functions of the state
    of the system at each time step

"""
function sdp_optimize(model::SPModel,
                  param::SDPparameters,
                  display=true::Bool)

    TF = model.stageNumber
    x = zeros(Float64, model.dimStates)
    x1 = zeros(Float64, model.dimStates)
    law = model.noises

    u_bounds = model.ulim
    x_bounds = model.xlim
    x_steps = param.stateSteps

    count_iteration = 1

    #Compute cartesian product spaces
    product_states, product_controls = generate_grid(model, param)

    V = zeros(Float64, param.stateVariablesSizes..., TF)

    #Compute final value functions
    for x in product_states
        indx = index_from_variable(x, x_bounds, x_steps)
        V[indx..., TF] = model.finalCostFunction(x)
    end

    #Construct a progress meter
    p = Progress((TF-1)*param.totalStateSpaceSize, 1)

    #Display start of the algorithm in DH and HD cases
    #Define the loops order in sdp
    if (param.infoStructure == "DH")
        if display
            println("Starting stochastic dynamic programming
                    decision hazard computation")
        end

        #Loop over time
        for t = (TF-1):-1:1
            count_iteration = count_iteration + 1
            Vtp1 = value_function_interpolation(model, V, t+1)

            #Loop over states
            for x in product_states

                if display
                    next!(p)
                end

                v = Inf
                v1 = 0
                u1 = tuple()
                Lv = 0

                #Loop over controls
                for u = product_controls

                    v1 = 0
                    count = 0

                    #Loop over uncertainty samples
                    for w = 1:param.monteCarloSize

                        wsample = sampling( law, t)
                        x1 = model.dynamics(t, x, u, wsample)

                        if model.constraints(t, x1, u, wsample)

                            count = count + 1
                            indx1 = real_index_from_variable(x1, x_bounds, x_steps)
                            itV = Vtp1[indx1...]
                            Lv = model.costFunctions(t, x, u, wsample)
                            v1 += Lv + itV

                        end
                    end

                    if (count>0)

                        v1 = v1 / count

                        if (v1 < v)

                            v = v1
                            u1 = u

                        end
                    end
                end
                indx = index_from_variable(x, x_bounds, x_steps)

                V[indx..., t] = v
            end
        end

    else
        if display
            println("Starting stochastic dynamic programming
                    hazard decision computation")
        end

        #Loop over time
        for t = (TF-1):-1:1
            count_iteration = count_iteration + 1
            Vtp1 = value_function_interpolation(model, V, t+1)

            #Loop over states
            for x in product_states

                if display
                    next!(p)
                end

                v     = 0
                Lv    = 0

                count = 0
                #Loop over uncertainty samples
                for w in 1:param.monteCarloSize

                    admissible_u_w_count = 0
                    v_x_w = 0
                    v_x_w1 = Inf
                    wsample = sampling( law, t)

                    #Loop over controls
                    for u in product_controls

                        x1 = model.dynamics(t, x, u, wsample)

                        if model.constraints(t, x1, u, wsample)

                            if (admissible_u_w_count == 0)
                                admissible_u_w_count = 1
                            end

                            Lv = model.costFunctions(t, x, u, wsample)
                            indx1 = real_index_from_variable(x1, x_bounds, x_steps)
                            itV = Vtp1[indx1...]
                            v_x_w1 = Lv + itV

                            if (v_x_w1 < v_x_w)
                                v_x_w = v_x_w1
                            end

                        end
                    end

                    v = v + v_x_w
                    count += 1
                end

                if (count>0)
                    v = v / count
                end
                indx = index_from_variable(x, x_bounds, x_steps)
                V[indx..., t] = v
            end
        end
    end

    return V
end


"""
Simulation of optimal control given an initial state and an alea scenario

Parameters:
- model (SPmodel)
    the DPSPmodel of our problem

- param (SDPparameters)
    the parameters for the SDP algorithm

- scenario (Array)
    the scenario of uncertainties realizations we want to simulate on

- X0 (SDPparameters)
    the initial state of the system

- value_functions (Array)
    the vector representing the value functions as functions of the state
    of the system at each time step

- display (Bool)
    the output display or verbosity parameter

Returns :

- J (Float)
    the cost of the optimal control over the scenario provided

- stocks (Array)
    the state of the controlled system at each time step

- controls (Array)
    the controls applied to the system at each time step

"""
function sdp_forward_simulation(model::SPModel,
                  param::SDPparameters,
                  scenario::Array,
                  X0::Array,
                  value::Array,
                  display=true::Bool)

    TF = model.stageNumber
    law = model.noises
    u_bounds = model.ulim
    x_bounds = model.xlim
    x_steps = param.stateSteps

    #Compute cartesian product spaces
    product_states, product_controls = generate_grid(model, param)

    controls = Inf*ones(TF-1, 1, model.dimControls)
    states = Inf*ones(TF, 1, model.dimStates)

    inc = 0
    for xj in X0
        inc = inc + 1
        states[1, 1, inc] = xj
    end

    J = 0
    xRef = X0

    if (param.infoStructure == "DH")
        #Decision hazard forward simulation
        for t = 1:(TF-1)

            x = states[t,1,:]

            uRef = tuple()

            LvRef = Inf
            Vtp1 = value_function_interpolation(model, value, t+1)

            for u in product_controls

                countW = 0.
                Lv = 0.
                for w = 1:param.monteCarloSize

                    wsample = sampling( law, t)

                    x1 = model.dynamics(t, x, u, wsample)

                    if model.constraints(t, x1, u, scenario[t])
                        indx1 = real_index_from_variable(x1, x_bounds, x_steps)
                        itV = Vtp1[indx1...]
                        Lv = Lv + model.costFunctions(t, x, u, scenario[t]) + itV
                        countW = countW +1.
                    end
                end
                Lv = Lv/countW
                if (Lv < LvRef)&(countW>0)
                    uRef = u
                    xRef = model.dynamics(t, x, u, scenario[t,1,:])
                    LvRef = Lv
                end
            end

            inc = 0

            for uj in uRef
                inc = inc +1
                controls[t,1,inc] = uj
            end

            inc = 0
            for xj in xRef
                inc = inc +1
                states[t+1,1,inc] = xj
            end

            J = J + model.costFunctions(t, x, uRef, scenario[t,1,:])

        end


    else
        #Hazard desision forward simulation
        for t = 1:TF-1

            x = states[t,1,:]

            Vtp1 = value_function_interpolation(model, value, t+1)

            uRef = tuple()

            xRef = tuple()
            LvRef = Inf

            for u = product_controls

                x1 = model.dynamics(t, x, u, scenario[t,1,:])

                if model.constraints(t, x1, u, scenario[t])
                    indx1 = real_index_from_variable(x1, x_bounds, x_steps)
                    itV = Vtp1[indx1...]
                    Lv = model.costFunctions(t, x, u, scenario[t,1,:]) + itV
                    if (Lv < LvRef)
                        uRef = u
                        xRef = model.dynamics(t, x, u, scenario[t,1,:])
                        LvRef = Lv
                    end
                end

            end

            inc = 0

            for uj in uRef
                inc = inc +1
                controls[t,1,inc] = uj
            end

            inc = 0
            for xj in xRef
                inc = inc +1
                states[t+1,1,inc] = xj
            end

            J = J + model.costFunctions(t, x, uRef, scenario[t])

        end
    end

    x = states[TF,1,:]
    J = J + model.finalCostFunction(x)

    return J, states, controls
end

