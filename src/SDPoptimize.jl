#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Stochastic dynamic programming algorithm
#
#############################################################################

function index_from_state( state::Array,
                    model::SPModel,
                    param::SDPparameters)
    index = 0;
    j = 1;

    for i = 1:model.dim_states
        index = index + j * int( ( state[i] - model.xlim[i][1] ) / param.stateSteps[i] )
        j = j * param.stateVariablesSizes[i]
    end

    return index
end

function index_from_control( control::Array,
                    model::SPModel,
                    param::SDPparameters)
    index = 0
    j = 1

    for i = 1:model.dim_controls
        index = index + j * int( ( control[i] - model.ulim[i][1] ) / param.controlSteps[i] )
        j = j * param.controlVariablesSizes[i]
    end

    return index
end

function state_from_index( state_ind::Int,
                    model::SPModel,
                    param::SDPparameters)

    state = zeros(model.dim_states)

    if (state_ind>-1):
        for i=1:model.dim_states
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

    control = zeros(model.dim_controls)

    if (control_ind > -1):
        for i=1:model.dim_controls
            control[i] = model.ulim[i][1] + param.controlSteps[i] * (index % param.controlVariablesSizes[i]);
            index = index -index % param.controlVariablesSizes[i];
            index = index / param.controlVariablesSizes[i];
        end
    end

    return control
end


function sdp_dh_value_iteration(model::SPModel,
                  param::SDPparameters,
                  display=true::Bool)

    V = zeros(Float64, model.stageNumber * param.stateSpaceSize)
    Pi = zeros(Float64, model.stageNumber * param.stateSpaceSize)
    x=zeros(Float64, model.dim_states)
    x1=zeros(Float64, model.dim_states)
    TF = model.stageNumber

    for indx = 1:param.stateSpaceSize
        x = state_from_index(indx,
                        model,
                        param)
        V[TF + (TF+1) * indx] = model.finalCostFunction(x)
    end

    for t = 1:TF

        for indx = 1:param.stateSpaceSize

            v = Inf
            v1 = Inf
            indu1 = -1
            Lv = 0
            x = state_from_index(indx)

            for indu = 1:param.controlSpaceSize

                v1 = 0
                count = 0

                for w = 1:param.monteCarloSize

                    law = [ model.noises[w] ]
                    u = control_from_index(indu)
                    wsample = simulate(law, 1)
                    x1 = model.dynamic(t, x, u, wsample)

                    if model.constraints(t, x1, x, u, wsample)

                        count = count + 1
                        indx1 = index_from_state(x1)
                        Lv = model.costFunctions(t, x, u, wsample)
                        v1 = v1 + Lv + V[t + 1 + (TF+1) * indx1]

                    end
                end

                if (count>0) :

                    v1 = v1 / count

                    if (v1 < v) :

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
                  display=true::Bool,
                  scenario::Array,
                  X0::Array,
                  value::Array,
                  policy::Array)

    TF = model.stageNumber

    U = zeros(TF)
    X = zeros(TF+1)
    X[1] = index_from_state(X0, model, param)
    J = value[1 + (TF+1) * X[1]]

    for t = 1:TF

        indx = X[t]
        x = state_from_index(indx, model, param)
        indu = policy[t + (TF+1) * indx]
        u = control_from_index(indu, model, param)

        x1 = model.dynamic(t, x, u, scenario[t])
        X[t+1] = index_from_state(x1, model, param)

    end

    return J, X

end