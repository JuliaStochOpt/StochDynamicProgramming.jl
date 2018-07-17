#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Definition of SDDP parameters
#############################################################################

mutable struct SDDPparameters
    # Solver used to solve LP
    SOLVER::MathProgBase.AbstractMathProgSolver
    # Solver used to solve MILP (default is nothing):
    MIPSOLVER::Nullable{MathProgBase.AbstractMathProgSolver}
    # number of scenarios in the forward pass
    forwardPassNumber::Int64
    # max iterations
    max_iterations::Int64
    # tolerance upon confidence interval:
    confidence_level::Float64
    # Estimate upper-bound every %% iterations:
    compute_ub::Int64
    # Number of MonteCarlo simulation to perform to estimate upper-bound:
    monteCarloSize::Int64
    # Number of MonteCarlo simulation to estimate the upper bound during one iteration
    in_iter_mc::Int64
    # Refresh JuMP Model:
    reload::Int
    # Pruning:
    prune::Bool

    function SDDPparameters(solver; passnumber=10, gap=0., confidence=.975,
                            max_iterations=20, prune_cuts=0,
                            pruning_algo="none",
                            compute_ub=-1, montecarlo_final=1000, montecarlo_in_iter=100,
                            mipsolver=nothing,
                            rho0=0., alpha=1., reload=-1, prune=false)

        return new(solver, mipsolver, passnumber, max_iterations, confidence,
                   compute_ub, montecarlo_final, montecarlo_in_iter, reload, prune)
    end
end


"""
Test compatibility of parameters.

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP
* `verbosity:Int64`:

# Return
`Bool`
"""
function check_SDDPparameters(model::SPModel, param::SDDPparameters, verbosity=0::Int64)
    if model.IS_SMIP && isnull(param.MIPSOLVER)
        error("MIP solver is not defined. Please set `param.MIPSOLVER`")
    end

    (verbosity > 0) && (model.IS_SMIP) && println("SMIP SDDP")
end

