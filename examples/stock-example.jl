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

######## Optimization parameters  ########
# choose the LP solver used. 
const SOLVER = ClpSolver()
# const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0)
# const SOLVER = GurobiSolver()

# convergence test
const EPSILON = 1e-3 #
const MAX_ITER = 100 # maximum iteration of SDDP

######## Stochastic Model  Parameters  ########
const N_STAGES = 5
const COSTS = rand(N_STAGES)

const CONTROL_MAX = 0.5
const CONTROL_MIN = 0

const XI_MAX = 0.3
const XI_MIN = 0
const N_XI = 10

const S0 = 0.5

# create law of noises
proba = 1/N_XI*ones(N_XI) # uniform probabilities
xi_support = collect(linspace(XI_MIN,XI_MAX,N_XI))
xi_law = NoiseLaw(xi_support, proba)
xi_laws = Vector{NoiseLaw}(N_STAGES)
for t=1:N_STAGES-1
   xi_laws[t] = xi_law
end

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
spmodel = LinearDynamicLinearCostSPmodel(N_STAGES,
                                                u_bounds,
                                                [S0],
                                                cost_t,
                                                dynamic,
                                                xi_laws)

set_state_bounds(spmodel, s_bounds)
paramSDDP = SDDPparameters(SOLVER, 10, 0, MAX_ITER) # 10 forward path, stop at MAX_ITER 

######### Solving the problem via SDDP
#V, pbs = solve_SDDP(spmodel, paramSDDP, 10) # display information every 10 iterations
#lb = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, V)
#println("Lower bound obtained by SDDP: "*string(lb))

######### Solving the problem via Dynamic Programming

stateSteps = [0.1]
controlSteps = [0.1]
infoStruct = "HD" # noise at time t is known before taking the decision at time t

paramSDP = SDPparameters(spmodel, stateSteps, controlSteps,  infoStruct)
V = sdp_optimize(spmodel,paramSDP)



######### Comparing the solution
#scenarios = generate_scenarios(xi_laws,1000) 
#costs, stocks = forward_simulations(spmodel, params, V, pbs, scenarios)

