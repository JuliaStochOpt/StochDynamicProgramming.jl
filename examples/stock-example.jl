#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Compare different ways of solving a stock problem :
# Min   E [\sum_{t=1}^TF c_t u_t]
# s.t.    s_{t+1} = s_t + u_t - xi_t, s_0 given
#         0 <= s_t <= 1
#         u_min <= u_t <= u_max
#         u_t choosen knowing xi_1 .. xi_t
#############################################################################
push!(LOAD_PATH, "../src")
using StochDynamicProgramming, JuMP, Clp, Distributions
println("library loaded")

run_sddp = true
run_sdp = true

######## Optimization parameters  ########
# choose the LP solver used.
const SOLVER = ClpSolver()
# const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0) # require "using CPLEX"

# convergence test
const MAX_ITER = 100 # maximum iteration of SDDP

######## Stochastic Model  Parameters  ########
const N_STAGES = 5
const COSTS = rand(N_STAGES)

const CONTROL_MAX = 0.5
const CONTROL_MIN = 0

const XI_MAX = 0.3
const XI_MIN = 0
const N_XI = 10
# initial stock
const S0 = 0.5

# create law of noises
proba = 1/N_XI*ones(N_XI) # uniform probabilities
xi_support = collect(linspace(XI_MIN,XI_MAX,N_XI))
xi_law = NoiseLaw(xi_support, proba)
xi_laws = NoiseLaw[xi_law for t in 1:N_STAGES-1]

# Define dynamic of the stock:
function dynamic(t, x, u, xi)
    return [x[1] + u[1] - xi[1]]
end

# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COSTS[t] * u[1]
end

######## Setting up the SPmodel
    s_bounds = [(0, 1)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX)]
    spmodel = LinearDynamicLinearCostSPmodel(N_STAGES,u_bounds,[S0],cost_t,dynamic,xi_laws)
    set_state_bounds(spmodel, s_bounds)


######### Solving the problem via SDDP
if run_sddp
    paramSDDP = SDDPparameters(SOLVER, 2, 0, MAX_ITER) # 10 forward pass, stop at MAX_ITER
    V, pbs = solve_SDDP(spmodel, paramSDDP, 10) # display information every 10 iterations
    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, V)
    println("Lower bound obtained by SDDP: "*string(lb_sddp))
end

######### Solving the problem via Dynamic Programming
if run_sdp
    stateSteps = [0.01]
    controlSteps = [0.01]
    infoStruct = "HD" # noise at time t is known before taking the decision at time t

    paramSDP = SDPparameters(spmodel, stateSteps, controlSteps, infoStruct)
    Vs = sdp_optimize(spmodel,paramSDP)
    lb_sdp = StochDynamicProgramming.get_value(spmodel,paramSDP,Vs)
    println("Value obtained by SDP: "*string(lb_sdp))
end

######### Comparing the solution
scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws,1000)
if run_sddp
    costsddp, stocks = forward_simulations(spmodel, paramSDDP, V, pbs, scenarios)
end
if run_sdp
    costsdp, states, stocks =sdp_forward_simulation(spmodel,paramSDP,scenarios,Vs)
end
if run_sddp && run_sdp
    println(mean(costsddp-costsdp))
end
