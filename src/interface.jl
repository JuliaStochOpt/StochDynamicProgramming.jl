
type SDDPInterface
    init::Bool
    # Stochastic model to solve
    spmodel::SPModel
    # SDDP parameters
    params::SDDPparameters
    # statistics
    stats::SDDPStat
    # cut pruner:
    pruner::Vector{CutPruners.AbstractCutPruner}

    # solution
    bellmanfunctions::Vector{PolyhedralFunction}
    solverinterface::Vector{JuMP.Model}

    verbose::Int

    # Init SDDP interface
    function SDDPInterface(model::SPModel, # SP Model
                           param::SDDPparameters; # parameters
                           verbose::Int=1)
        check_SDDPparameters(model, param, verbose)
        # initialize value functions:
        V, problems = initialize_value_functions(model, param)
        (verbose > 0) && println("SDDP Interface initialized")

        pruner = initpruner(param, model.stageNumber, model.dimStates)
        #Initialization of stats
        stats = SDDPStat()
        return new(false, model, param, stats, pruner, V, problems, verbose)
    end
    function SDDPInterface(model::SPModel,
                        params::SDDPparameters,
                        V::Vector{PolyhedralFunction};
                        verbose::Int=1)
        check_SDDPparameters(model, params, verbose)
        # First step: process value functions if hotstart is called
        problems = hotstart_SDDP(model, params, V)
        pruner = initpruner(params, model.stageNumber, model.dimStates)
        stats = SDDPStat()
        return new(false, model, params, stats, pruner, V, problems, verbose)
    end
end



function initpruner(param, nstages, ndim)
    algo = param.pruning[:algo]
    # Initialize cuts container for cuts pruning:
    return [CutPruners.CutPruner{ndim, Float64}(algo) for i in 1:nstages-1]
end
