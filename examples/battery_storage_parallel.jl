#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Solving an decision hazard battery storage problem using multiprocessed 
# dynamic programming :
# We manage a network connecting an electrical demand, 
# a renewable energy production unit, a battery and the global network.
# We assume electrical demand d_t as well as cost of electricity c_t deterministic 
# We decide which quantity to store before knowing renewable energy production
# If more energy comes, we store the excess up to the state of charge upper bound
# The remaining excess is wasted
# If not enough energy comes, we lower accordingly what was decided to discharge
# to ensure state of charge lower bound constraint
# We have to ensure supply/demand balance: the energy provided by the network
# equals the demand minus the renewable production, plus the battery demand 
# or minus the battery production: G_t = max(d_t - xi_t, 0) + F_t(u_t)
# We forbid electricity sale to the network
# Min   E [\sum_{t=1}^TF c_t G_t]
# s.t.    s_{t+1} = s_t - u_t + max(0,xi_t-d_t), if 0 <= s_t - u_t + max(0,xi_t-d_t) <= s_{max}
#         s_{t+1} = s_{max}, if s_t - u_t + max(0,xi_t-d_t) >= s_{min}
#         s_{t+1} = 0, if s_t - u_t + max(0,xi_t-d_t) < 0
#         F_t(u_t) = max(0,s_{max} - s_{t} - max(0,xi_t-d_t)), if s_{t+1} = s_{max} 
#		  F_t(u_t) = s_{min} - s_{t} - max(0,xi_t-d_t), if s_{t+1} = s_{min}
#         F_t(u_t) = u_t, otherwise
#         s_0 given
#         s_{min} <= s_t <= s_{max}
#         u_min <= u_t <= u_max
#         u_t choosen knowing xi_0 .. xi_{t-1}
#############################################################################

import StochDynamicProgramming, Distributions
println("library loaded")

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

    # create law of noises
    proba = 1/N_XI*ones(N_XI) # uniform probabilities
    xi_support = collect(linspace(XI_MIN,XI_MAX,N_XI))
    xi_law = StochDynamicProgramming.NoiseLaw(xi_support, proba)
    xi_laws = StochDynamicProgramming.NoiseLaw[xi_law for t in 1:N_STAGES-1]

    # Define dynamic of the stock:
    function dynamic(t, x, u, xi)
        return [min(STATE_MAX, max(STATE_MIN,x[1] + u[1] + max(0,xi[1], DEMAND[t])))]
    end

    # Define cost corresponding to each timestep:
    function cost_t(t, x, u, xi)
    	x1 = dynamic(t, x, u, xi)[1]
		c = max(0, DEMAND[t] - xi[1])
    	if x1 == STATE_MAX
    		c += max(0, STATE_MAX - x[1] - max(0,xi[1], DEMAND[t]))
    	elseif x1 == STATE_MIN
    		c += STATE_MIN - x[1] - max(0,xi[1], DEMAND[t])
    	else
    		c += u[1] 
    	end 
        return COSTS[t] * c
    end

    ######## Setting up the SPmodel
    s_bounds = [(STATE_MIN, STATE_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX)]
    spmodel = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(N_STAGES,
                                                                    u_bounds,
                                                                    [S0],
                                                                    cost_t,
                                                                    dynamic,
                                                                    xi_laws)
    StochDynamicProgramming.set_state_bounds(spmodel, s_bounds)

    scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws,1000)

    stateSteps = [0.01]
    controlSteps = [0.001]
    infoStruct = "DH" # noise at time t is not known before taking the decision at time t

    paramSDP = StochDynamicProgramming.SDPparameters(spmodel, stateSteps,
                                                    controlSteps, infoStruct)
end

Vs = []

@time for i in 1:1
Vs = StochDynamicProgramming.solve_DP(spmodel,paramSDP, 1)
end

lb_sdp = StochDynamicProgramming.get_bellman_value(spmodel,paramSDP,Vs)
println("Value obtained by SDP: "*string(lb_sdp))
costsdp, states, stocks = StochDynamicProgramming.sdp_forward_simulation(spmodel,paramSDP,scenarios,Vs)
println(mean(costsdp))

