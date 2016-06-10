#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Compare the optimal values and control returned by different instances
#  of SDDP on the same problem
#############################################################################

"""
Create different sets of parameters from a problem and compare the results
of these different instances.

"""
function benchmark_parameters(model,
          SDDParametersCollection,
          seeds, scenarios)

    for i in 1:length(SDDParametersCollection)

        srand(seeds)

        (V, pbs, ic), t1, m1 = @timed solve_SDDP(model, SDDParametersCollection[i], 0)
        lb_sddp, t2, m2 = @timed StochDynamicProgramming.get_lower_bound(model, SDDParametersCollection[i], V)
        (costsddp, stocks), t3, m3 = @timed forward_simulations(model, SDDParametersCollection[i], V, pbs, scenarios)

        V0, t4, m4 = @timed get_bellman_value(model, SDDParametersCollection[i], 1, V[1], model.initialState)
        (upb, costs), t5, m5 = @timed estimate_upper_bound(model, SDDParametersCollection[i], V, pbs)

        time = t1+t2+t3+t4+t5
        memory = m1+m2+m3+m4+m5

        print("Instance ",i,"\t")
        print("time = ",round(time,4),"\t")
        print("memory = ",memory,"\t")
        print("gap = ", round(100*(upb-V0)/V0),"\t")
        print("iteration count = ", ic,"\t")
        println("number CPLEX call = ", ic*(model.stageNumber-1)*(SDDParametersCollection[i].forwardPassNumber+1))
    end
end
