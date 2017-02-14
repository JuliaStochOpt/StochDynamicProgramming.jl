
# Define alias for cuts pruning algorithm:
typealias LevelOne Val{:LevelOne}
typealias ExactPruning Val{:Exact}
typealias Territory Val{:Exact_Plus}
typealias NoPruning Val{:none}

type SDDPparameters
    # Solver used to solve LP
    SOLVER::MathProgBase.AbstractMathProgSolver
    # Solver used to solve MILP (default is nothing):
    MIPSOLVER::Union{Void, MathProgBase.AbstractMathProgSolver}
    # number of scenarios in the forward pass
    forwardPassNumber::Int64
    # Admissible gap between lower and upper-bound:
    gap::Float64
    # tolerance upon confidence interval:
    confidence_level::Float64
    # Maximum iterations of the SDDP algorithms:
    maxItNumber::Int64
    # Define the pruning method
    pruning::Dict{Symbol, Any}
    # Estimate upper-bound every %% iterations:
    compute_ub::Int64
    # Number of MonteCarlo simulation to perform to estimate upper-bound:
    monteCarloSize::Int64
    # Number of MonteCarlo simulation to estimate the upper bound during one iteration
    in_iter_mc::Int64
    # specify whether SDDP is accelerated
    IS_ACCELERATED::Bool
    # ... and acceleration parameters:
    acceleration::Dict{Symbol, Float64}

    function SDDPparameters(solver; passnumber=10, gap=0., confidence=.975,
                            max_iterations=20, prune_cuts=0,
                            pruning_algo="none",
                            compute_ub=-1, montecarlo_final=1000, montecarlo_in_iter=100,
                            mipsolver=nothing,
                            rho0=0., alpha=1.)

        pruning_algo = CutPruners.AvgCutPruningAlgo(-1)
        is_acc = (rho0 > 0.)
        accparams = is_acc? Dict(:ρ0=>rho0, :alpha=>alpha, :rho=>rho0): Dict()

        prune_cuts = (pruning_algo != "none")? prune_cuts: 0

        corresp = Dict("none"=>NoPruning,
                       "level1"=>LevelOne,
                       "exact+"=>Territory,
                       "exact"=>ExactPruning)
        prune_cuts = Dict(:pruning=>prune_cuts>0,
                          :period=>prune_cuts,
                          :algo=>pruning_algo)
        return new(solver, mipsolver, passnumber, gap, confidence,
                   max_iterations, prune_cuts, compute_ub,
                   montecarlo_final, montecarlo_in_iter, is_acc, accparams)
    end
end

"""
Test compatibility of parameters.

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP
* `verbose:Int64`:

# Return
`Bool`
"""
function check_SDDPparameters(model::SPModel, param::SDDPparameters, verbose=0::Int64)
    if model.IS_SMIP && isa(param.MIPSOLVER, Void)
        error("MIP solver is not defined. Please set `param.MIPSOLVER`")
    end
    (model.IS_SMIP && param.IS_ACCELERATED) && error("Acceleration of SMIP not supported")

    (verbose > 0) && (model.IS_SMIP) && println("SMIP SDDP")
    (verbose > 0) && (param.IS_ACCELERATED) && println("Acceleration: ON")
end

