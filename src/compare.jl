#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Compare the optimal values and control returned by different instances
#  of SDDP on the same problem
#############################################################################

"""
Compare different sets of parameters to solve an instance of SDDP

# Description
Take a collection of SDDP parameters and compare the time of execution,
the memory used, an estimation of the gap to optimality and the number 
of calls to the solver

# Arguments
* `model::SPmodel`:
    The stochastic problem we want to benchmark
* `SDDParametersCollection::Array{Any,1}`
    Collection of SDDPparameters
* `scenarios::Array{Float64,3}`
    Set of scenarios used to calculate costs
* `seeds::Int`
    The random number generator seeds
    
# Output
* `Display in the terminal`
    Print information in the terminal
"""
function benchmark_parameters(model,
                               SDDParametersCollection,
                               scenarios::Array{Float64,3},
                               seeds::Int)

    #Execute a first time each function to compile them
    (V, pbs, callsolver), t1, m1 = @timed solve_SDDP(model, SDDParametersCollection[1], 0)
     V0, t2, m2 = @timed get_bellman_value(model, SDDParametersCollection[1], 1, V[1], model.initialState)
    (upb, costs), t3, m3 = @timed estimate_upper_bound(model, SDDParametersCollection[1], scenarios, pbs)


    for sddpparams in SDDParametersCollection

        srand(seeds)

        (V, pbs, callsolver), t1, m1 = @timed solve_SDDP(model, sddpparams, 0)
        V0, t2, m2 = @timed get_bellman_value(model, sddpparams, 1, V[1], model.initialState)
        (upb, costs), t3, m3 = @timed estimate_upper_bound(model, sddpparams, scenarios, pbs)

        solvingtime = t1
        simulationtime = t2+t3
        solvingmemory = m1
        simulationmemory = m2+m3
        
        print("Instance \t")
        print("Solving time = ",round(solvingtime,4),"\t")
        print("Solving memory = ", solvingmemory,"\t")
        print("Simulation time = ",round(simulationtime,4),"\t")
        print("Simulation memory = ", simulationmemory,"\t")
        print("Gap < ", round(100*(upb-V0)/V0),"% with prob 97.5%\t")
        println("number external solver call = ", callsolver)
    end
end
