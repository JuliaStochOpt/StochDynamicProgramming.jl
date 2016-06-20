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

Parameters:
- model (SPmodel)
    the stochastic problem we want to benchmark

- SDDParametersCollection ()

- seeds (Int)
    The seed which is chosen by the users to compare parameters on the same aleas

- scenarios ()
    Set of scenarios used to calculate costs

"""
function benchmark_parameters(model,
          SDDParametersCollection,
          seeds, scenarios)

    #Execute a first time each function to compile them
    (V, pbs, callsolver), t1, m1 = @timed solve_SDDP(model, SDDParametersCollection[1], 0)
    lb_sddp, t2, m2 = @timed StochDynamicProgramming.get_lower_bound(model, SDDParametersCollection[1], V)
    (costsddp, stocks,_), t3, m3 = @timed forward_simulations(model, SDDParametersCollection[1], pbs, scenarios)

    V0, t4, m4 = @timed get_bellman_value(model, SDDParametersCollection[1], 1, V[1], model.initialState)
    (upb, costs), t5, m5 = @timed estimate_upper_bound(model, SDDParametersCollection[1], V, pbs)


    for sddpparams in SDDParametersCollection

        srand(seeds)

        (V, pbs, callsolver), t1, m1 = @timed solve_SDDP(model, sddpparams, 0)
        lb_sddp, t2, m2 = @timed StochDynamicProgramming.get_lower_bound(model, sddpparams, V)
        (costsddp, stocks,_), t3, m3 = @timed forward_simulations(model, sddpparams, pbs, scenarios)

        V0, t4, m4 = @timed get_bellman_value(model, sddpparams, 1, V[1], model.initialState)
        (upb, costs), t5, m5 = @timed estimate_upper_bound(model, sddpparams, V, pbs)

        time = t1+t2+t3+t4+t5
        memory = m1+m2+m3+m4+m5

        print("Instance \t")
        print("time = ",round(time,4),"\t")
        print("memory = ",memory,"\t")
        print("gap = ", round(100*(upb-V0)/V0),"\t")
        println("number CPLEX call = ", callsolver)
    end
end
