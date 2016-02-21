#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Stochastic dynamic programming algorithm
#
#############################################################################

using ProgressMeter

function index_from_state( state::Array,
                    model::SPModel,
                    param::SDPparameters)
    index = 0;
    j = 1;

    for i = 1:model.dimStates
        index = index + j * round( Int, ( state[i] - model.xlim[i][1] ) / param.stateSteps[i] )
        j = j * param.stateVariablesSizes[i]
    end

    return index
end

function index_from_control( control::Array,
                    model::SPModel,
                    param::SDPparameters)
    index = 0
    j = 1

    for i = 1:model.dimControls
        index = index + j * round( Int, ( control[i] - model.ulim[i][1] ) / param.controlSteps[i] )
        j = j * param.controlVariablesSizes[i]
    end

    return index
end

function state_from_index( state_ind::Int,
                    model::SPModel,
                    param::SDPparameters)
    index = state_ind
    state = zeros(model.dimStates)

    if (state_ind>-1)
        for i=1:model.dimStates
            state[i] = model.xlim[i][1] + param.stateSteps[i] * (index % param.stateVariablesSizes[i])
            index = index -index % param.stateVariablesSizes[i]
            index = index / param.stateVariablesSizes[i]
        end
    end

    return state
end

function control_from_index( control_ind::Int,
                    model::SPModel,
                    param::SDPparameters)
    index = control_ind
    control = zeros(model.dimControls)

    if (control_ind > -1)
        for i=1:model.dimControls
            control[i] = model.ulim[i][1] + param.controlSteps[i] * (index % param.controlVariablesSizes[i]);
            index = index -index % param.controlVariablesSizes[i];
            index = index / param.controlVariablesSizes[i];
        end
    end

    return control
end


function sdp_optimize(model::SPModel,
                  param::SDPparameters,
                  display=true::Bool)

    V = zeros(Float64, (model.stageNumber+1) * param.totalStateSpaceSize)
    Pi = zeros(Int64, (model.stageNumber+1) * param.totalStateSpaceSize)
    x=zeros(Float64, model.dimStates)
    x1=zeros(Float64, model.dimStates)
    TF = model.stageNumber
    law = model.noises

    count_iteration = 1
    println("Iteration number : ", count_iteration)
    for indx = 0:(param.totalStateSpaceSize-1)
        x = state_from_index(indx,
                        model,
                        param)
        V[TF+1 + (TF+1) * indx] = model.finalCostFunction(x)
    end

    #p = Progress(TF*100*100, 1)
    for t = 1:TF
        count_iteration = count_iteration + 1
        println("Iteration number : ", count_iteration)

        for indx = 0:(param.totalStateSpaceSize-1)

            v = Inf
            v1 = Inf
            indu1 = -1
            Lv = 0
            x = state_from_index(indx, model, param)

            for indu = 0:(param.totalControlSpaceSize-1)
                #next!(p)

                v1 = 0
                count = 0

                for w = 1:param.monteCarloSize

                    u = control_from_index(indu, model, param)
                    wsample = law[t].support[:, rand(Categorical(law[t].proba))]
                    x1 = model.dynamics(t, x, u, wsample)

                    if model.constraints(t, x, x1, u, wsample)

                        count = count + 1
                        indx1 = index_from_state(x1, model, param)
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




function sdp_forward_simulation(model::SPModel,
                  param::SDPparameters,
                  scenario::Array,
                  X0::Array,
                  value::Array,
                  policy::Array,
                  display=true::Bool)

    TF = model.stageNumber

    U = zeros(Int64, TF)
    X = zeros(Int64, TF+1)
    X[1] = index_from_state(X0, model, param)
    J = value[1 + (TF+1) * X[1]]

    for t = 1:TF

        indx = X[t]
        x = state_from_index(indx, model, param)
        indu = policy[t + (TF+1) * indx]
        u = control_from_index(indu, model, param)

        x1 = model.dynamics(t, x, u, scenario[t])
        X[t+1] = index_from_state(x1, model, param)

    end

    return J, X
end