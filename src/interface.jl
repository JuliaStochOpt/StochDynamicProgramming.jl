
type SDDPInterface
    init::Bool
    # Stochastic model to solve
    spmodel::SPModel
    # SDDP parameters
    params::SDDPparameters
    # statistics
    stats::SDDPStat

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

        #Initialization of stats
        stats = SDDPStat()
        return new(false, model, param, stats, V, problems, verbose)
    end
end

function SDDPInterface(model::SPModel,
                       params::SDDPparameters,
                       V::Vector{PolyhedralFunction};
                       verbose::Int=1)
    check_SDDPparameters(model, param, verbose)
    # First step: process value functions if hotstart is called
    problems = hotstart_SDDP(model, param, V)
    return SDDPInterface(false, model, params, stats, V, problems, verbose)
end
