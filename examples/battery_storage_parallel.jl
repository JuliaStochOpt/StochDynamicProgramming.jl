#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Solving an decision hazard battery storage problem using multiprocessed
# dynamic programming :
# We manage a network connecting an electrical demand,
# a renewable energy production unit, a battery and the global network.
# We want to minimize the cost of consumed electricity until time T: \sum_{t=0}^T c_t * G_{t+1}.
# We assume electrical demand d_t as well as cost of electricity c_t deterministic
# We decide which quantity to store before knowing renewable energy production
# but we don't waste the eventual excess.
# If more energy comes, we store the excess up to the state of charge upper bound.
# The remaining excess is provided directly to the demand or wasted if still too important.
# We have to ensure supply/demand balance: the energy provided by the network G_{t+1}
# equals the demand d_t plus the battery effective demand or minus the battery effective
# production U_{t+1} then minus the used renewable production xi_{t+1} - R_{t+1].
# R_{t+1] is the renewable energy wasted/curtailed
# U_{t+1} is a function of the decision variable: the amount of energy decided
# to store or discharge u_t before the uncertainty realization.
# We forbid electricity sale to the network: G_t+1 >=0 .
# This inequality constraint can be translated on an equality constraint on R_{t+1}
# Min   E [\sum_{t=1}^T c_t G_{t+1}]
# s.t.    s_{t+1} = s_t + U_{t+1}
#		  G_{t+1} = d_t + U_{t+1} - (W_{t+1} - R_{t+1})
#		  U_{t+1} = | rho_c * min(S_{max} - S_t, min( u_t ,W_{t+1})), if u_t >=0
#					| (1/rho_dc) * max(u_t, -max( d_t - W_{t+1}, 0)), otherwise
#         R_{t+1} = max(W_{t+1} - d_t - U_{t+1}, 0)
#         s_0 given
#         s_{min} <= s_t <= s_{max}
#         u_min <= u_t <= u_max
#         u_t choosen knowing xi_0 .. xi_{t-1}
#############################################################################

import StochDynamicProgramming, Distributions
println("library loaded")

# We have to define the instance on all the workers (processes)
@everywhere begin

    run_sdp = true

    ######## Stochastic Model  Parameters  ########
    const N_STAGES = 50
    const COSTS = rand(N_STAGES)
    const DEMAND = rand(N_STAGES)

    const CONTROL_MAX = 0.5
    const CONTROL_MIN = 0

    const STATE_MAX = 1
    const STATE_MIN = 0

    const XI_MAX = 0.3
    const XI_MIN = 0
    const N_XI = 10
    # initial stock
    const S0 = 0.5

    # charge and discharge efficiency parameters
    const rho_c = 0.98
    const rho_dc = 0.97

    # create law of noises
    proba = 1/N_XI*ones(N_XI) # uniform probabilities
    xi_support = collect(linspace(XI_MIN,XI_MAX,N_XI))
    xi_law = StochDynamicProgramming.NoiseLaw(xi_support, proba)
    xi_laws = StochDynamicProgramming.NoiseLaw[xi_law for t in 1:N_STAGES-1]

    # Define dynamic of the stock:
    function dynamic(t, x, u, xi)
    	if u[1]>=0
    		return [ x[1] + rho_c * min(max(u[1], xi[1]),STATE_MAX-x[1])]
    	else
    		return [ x[1] + 1/rho_dc * max(u[1],-max(0,DEMAND[t])) ]
    	end
    end

    # Define cost corresponding to each timestep:
    function cost_t(t, x, u, xi)
    	U = 0
    	if u[1]>=0
    		U = rho_c * min(max(u[1], xi[1]),STATE_MAX-x[1])
    	else
    		U = 1/rho_dc * max(u[1],-max(0,DEMAND[t]))
    	end
        return COSTS[t] * max(0, DEMAND[t] + U - xi[1])
    end

    function constraint(t, x, u, xi)
    	return( (x[1] <= s_bounds[1][2] )&(x[1] >= s_bounds[1][1]))
    end

    function finalCostFunction(x)
    	return(0)
    end

    ######## Setting up the SPmodel
    s_bounds = [(STATE_MIN, STATE_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX)]
    spmodel = StochDynamicProgramming.StochDynProgModel(N_STAGES, s_bounds,
                                                                    u_bounds,
                                                                    [S0],
                                                                    cost_t,
                                                                    finalCostFunction,
                                                                    dynamic,
                                                                    constraint,
                                                                    xi_laws)

    scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws,1000)

    stateSteps = [0.01]
    controlSteps = [0.001]
    infoStruct = "DH" # noise at time t is not known before taking the decision at time t

    paramSDP = StochDynamicProgramming.SDPparameters(spmodel, stateSteps,
                                                    controlSteps, infoStruct)
end

Vs = StochDynamicProgramming.solve_DP(spmodel,paramSDP, 1)

lb_sdp = StochDynamicProgramming.get_bellman_value(spmodel,paramSDP,Vs)
println("Value obtained by SDP: "*string(lb_sdp))
costsdp, states, stocks = StochDynamicProgramming.sdp_forward_simulation(spmodel,paramSDP,scenarios,Vs)
println(mean(costsdp))

