#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define all types used in this module.
#############################################################################


@compat abstract type SPModel end


type PolyhedralFunction
    #function defined by max_k betas[k] + lambdas[k,:]*x
    betas::Vector{Float64}
    lambdas::Array{Float64,2} #lambdas[k,:] is the subgradient
    # number of cuts:
    numCuts::Int64
    hashcuts::Vector{UInt64}
    newcuts::Int
end

PolyhedralFunction(n_dim::Int) = PolyhedralFunction(Float64[], Array{Float64}(0, n_dim), 0, UInt64[], 0)
PolyhedralFunction(beta, lambda) = PolyhedralFunction(beta, lambda, length(beta), UInt64[], 0)

function fetchnewcuts!(V::PolyhedralFunction)
    β = V.betas[end-V.newcuts+1:end]
    λ = V.lambdas[end-V.newcuts+1:end, :]
    V.newcuts = 0
    return β, λ
end

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
    equalityConstraints::Nullable{Function}
    inequalityConstraints::Nullable{Function}
    info::Symbol

    IS_SMIP::Bool

    function LinearSPModel(n_stage,             # number of stages
                           u_bounds,            # bounds of control
                           x0,                 # initial state
                           cost,               # cost function
                           dynamic,            # dynamic
                           aleas;              # modelling of noises
                           Vfinal=nothing,     # final cost
                           eqconstr=nothing,   # equality constraints
                           ineqconstr=nothing, # inequality constraints
                           info=:HD,           # information structure
                           control_cat=nothing) # category of controls

        dimStates = length(x0)
        dimControls = length(u_bounds)
        dimNoises = length(aleas[1].support[:, 1])

        # First step: process terminal costs.
        # If not specified, default value is null function
        if isa(Vfinal, Function) || isa(Vfinal, PolyhedralFunction)
            Vf = Vfinal
        else
            Vf = PolyhedralFunction(zeros(1), zeros(1, dimStates), 1, UInt64[], 0)
        end

        isbu = isa(control_cat, Vector{Symbol})? control_cat: [:Cont for i in 1:dimControls]
        is_smip = (:Int in isbu)||(:Bin in isbu)

        x_bounds = [(-Inf, Inf) for i=1:dimStates]

        return new(n_stage, dimControls, dimStates, dimNoises, x_bounds, u_bounds,
                   x0, cost, dynamic, aleas, Vf, isbu, eqconstr, ineqconstr, info, is_smip)
    end
end


"""Set bounds on state."""
function set_state_bounds(model::SPModel, x_bounds)
    if length(x_bounds) != model.dimStates
        error("Bounds dimension, must be ", model.dimStates)
    else
        model.xlim = x_bounds
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

    build_search_space::Nullable{Function}

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
                                finalCostFunction, dynamic, constraints, aleas, search_space_builder = Nullable{Function}())
        return new(TF, length(u_bounds), length(x_bounds), length(aleas[1].support[:, 1]),
                    x_bounds, u_bounds, x0, cost_t, finalCostFunction, dynamic,
                    constraints, aleas, search_space_builder)
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


@compat abstract type AbstractSDDPStats end

# Define an object to store evolution of solution
# along iterations:
type SDDPStat <: AbstractSDDPStats
    # Number of iterations:
    niterations::Int
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
    # time used to solve each LP:
    solverexectime_fw::Vector{Float64}
    solverexectime_bw::Vector{Float64}
    # number of calls to solver:
    nsolved::Int
    # number of optimality cuts
    nocuts::Int
    npaths::Int
    # current lower bound
    lowerbound::Float64
    # current lower bound
    upperbound::Float64
    # upper-bound std:
    σ_UB::Float64
    # total time
    time::Float64
end


SDDPStat() = SDDPStat(0, [], [], [], [], [], [], [], 0, 0, 0, 0., 0., 0., 0.)

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
    stats.lowerbound = lwb
    stats.upperbound = upb[1]
    stats.σ_UB = upb[2]
    stats.time += time
end


type NLDSSolution
    # solver status:
    status::Bool
    # cost:
    objval::Float64
    # next position:
    xf::Array{Float64, 1}
    # optimal control:
    uopt::Array{Float64, 1}
    # Subgradient:
    ρe::Array{Float64, 1}
    # cost-to-go:
    θ::Float64
    # cuts' multipliers
    πc::Vector{Float64}
end

NLDSSolution() = NLDSSolution(false, Inf, Array{Float64, 1}(), Array{Float64, 1}(), Array{Float64, 1}(), Inf)
