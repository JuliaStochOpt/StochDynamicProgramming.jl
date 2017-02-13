#  Copyright 2014, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define all types used in this module.
#############################################################################


abstract SPModel


type PolyhedralFunction
    #function defined by max_k betas[k] + lambdas[k,:]*x
    betas::Vector{Float64}
    lambdas::Array{Float64,2} #lambdas[k,:] is the subgradient
    # number of cuts:
    numCuts::Int64
end

PolyhedralFunction(ndim) = PolyhedralFunction([], Array{Float64}(0, ndim), 0)


type LinearSPModel <: SPModel
    # problem dimension
    stageNumber::Int64
    dimControls::Int64
    dimStates::Int64
    dimNoises::Int64

    # Bounds of states and controls:
    xlim::Array{Tuple{Float64,Float64},1}
    ulim::Array{Tuple{Float64,Float64},1}

    initialState::Array{Float64, 1}

    #FIXME: add a correct typage for costFunctions that dont break in 0.5
    costFunctions
    dynamics::Function
    noises::Vector{NoiseLaw}

    finalCost::Union{Function, PolyhedralFunction}

    controlCat::Vector{Symbol}
    equalityConstraints::Union{Void, Function}
    inequalityConstraints::Union{Void, Function}

    refTrajectories::Union{Void, Array{Float64, 3}}

    IS_SMIP::Bool

    function LinearSPModel(nstage,             # number of stages
                           ubounds,            # bounds of control
                           x0,                 # initial state
                           cost,               # cost function
                           dynamic,            # dynamic
                           aleas;              # modelling of noises
                           Vfinal=nothing,     # final cost
                           eqconstr=nothing,   # equality constraints
                           ineqconstr=nothing, # inequality constraints
                           control_cat=nothing) # category of controls

        dimStates = length(x0)
        dimControls = length(ubounds)
        dimNoises = length(aleas[1].support[:, 1])

        # First step: process terminal costs.
        # If not specified, default value is null function
        if isa(Vfinal, Function) || isa(Vfinal, PolyhedralFunction)
            Vf = Vfinal
        else
            Vf = PolyhedralFunction(zeros(1), zeros(1, dimStates), 1)
        end

        isbu = isa(control_cat, Vector{Symbol})? control_cat: [:Cont for i in 1:dimStates]
        is_smip = (:Int in isbu)||(:Bin in isbu)

        xbounds = [(-Inf, Inf) for i=1:dimStates]

        return new(nstage, dimControls, dimStates, dimNoises, xbounds, ubounds,
                   x0, cost, dynamic, aleas, Vf, isbu, eqconstr, ineqconstr, nothing, is_smip)
    end
end


"""Set bounds on state."""
function set_state_bounds(model::SPModel, xbounds)
    if length(xbounds) != model.dimStates
        error("Bounds dimension, must be ", model.dimStates)
    else
        model.xlim = xbounds
    end
end


type StochDynProgModel <: SPModel
    # problem dimension
    stageNumber::Int64
    dimControls::Int64
    dimStates::Int64
    dimNoises::Int64

    # Bounds of states and controls:
    xlim::Array{Tuple{Float64,Float64},1}
    ulim::Array{Tuple{Float64,Float64},1}

    initialState::Array{Float64, 1}

    costFunctions::Function
    finalCostFunction::Function
    dynamics::Function
    constraints::Function
    noises::Vector{NoiseLaw}

    function StochDynProgModel(model::LinearSPModel, final, cons)
        if isa(model.costFunctions, Function)
            cost = model.costFunctions
        #FIXME: broken test since 0.5 release
        else
            function cost(t,x,u,w)
                current_cost = -Inf
                for aff_func in model.costFunctions
                    current_cost = aff_func(t,x,u,w)
                end
            return current_cost
            end
        end
        return StochDynProgModel(model.stageNumber, model.xlim, model.ulim, model.initialState,
                 cost, final, model.dynamics, cons,
                 model.noises)
    end

    function StochDynProgModel(TF, x_bounds, u_bounds, x0, cost_t,
                                finalCostFunction, dynamic, constraints, aleas)
        return new(TF, length(u_bounds), length(x_bounds), length(aleas[1].support[:, 1]),
                    x_bounds, u_bounds, x0, cost_t, finalCostFunction, dynamic,
                    constraints, aleas)
    end

end


type SDPparameters
    stateSteps
    controlSteps
    totalStateSpaceSize
    totalControlSpaceSize
    stateVariablesSizes
    controlVariablesSizes
    monteCarloSize
    infoStructure
    expectation_computation

    function SDPparameters(model, stateSteps, controlSteps, infoStruct,
                            expectation_computation="Exact" ,monteCarloSize=1000)

        stateVariablesSizes = zeros(Int64, length(stateSteps))
        controlVariablesSizes = zeros(Int64, length(controlSteps))
        totalStateSpaceSize = 1
        totalControlSpaceSize = 1
        for i=1:length(stateSteps)
            stateVariablesSizes[i] = round(Int64,1 +
                                    (model.xlim[i][2]-model.xlim[i][1])/stateSteps[i])
            totalStateSpaceSize *= stateVariablesSizes[i]
        end

        for i=1:length(controlSteps)
            controlVariablesSizes[i] = round(Int64, 1 +
                                    (model.ulim[i][2]-model.ulim[i][1])/controlSteps[i])
            totalControlSpaceSize *= controlVariablesSizes[i]
        end

        return new(stateSteps, controlSteps, totalStateSpaceSize,
                    totalControlSpaceSize, stateVariablesSizes,
                    controlVariablesSizes, monteCarloSize, infoStruct,
                    expectation_computation)
    end

end


# Define an object to store evolution of solution
# along iterations:
type SDDPStat
    # Number of iterations:
    niterations::Int64
    # evolution of lower bound:
    lower_bounds::Vector{Float64}
    # evolution of upper bound:
    upper_bounds::Vector{Float64}
    # standard deviation of upper-bound's estimation
    upper_bounds_std::Vector{Float64}
    # tolerance of upper-bounds estimation:
    upper_bounds_tol::Vector{Float64}
    # evolution of execution time:
    exectime::Vector{Float64}
    # number of calls to solver:
    ncallsolver::Int64
end

SDDPStat() = SDDPStat(0, [], [], [], [], [], 0)

"""
Update the SDDPStat object with the results of current iterations.

# Arguments
* `stats::SDDPStat`:
    statistics of the current algorithm
* `call_solver_at_it::Int64`:
    number of time a solver was called during the current iteration
* `lwb::Float64`:
    lowerbound obtained
* `upb::Float64`:
    upperbound estimated
* `time`
"""
function updateSDDPStat!(stats::SDDPStat,
                         lwb::Float64,
                         upb::Vector{Float64},
                         time)
    stats.niterations += 1
    push!(stats.lower_bounds, lwb)
    push!(stats.upper_bounds, upb[1])
    push!(stats.upper_bounds_tol, upb[3])
    push!(stats.upper_bounds_std, upb[2])
    push!(stats.exectime, time)
end


type NextStep
    next_state::Array{Float64, 1}
    optimal_control::Array{Float64, 1}
    sub_gradient
    cost::Float64
    cost_to_go::Float64
end

