#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Compare different ways of solving a stock problem :
# Min   E [\sum_{t=1}^TF \sum_{i=1}^N c^i_t u^i_t]
# s.t.    s^i_{t+1} = s^i_t + u^i_t - xi^i_t, s_0 given
#         0 <= s^i_t <= 1
#         \sum_{i=1}^N u^i_t <= rN
#         u_min <= u^i_t <= u_max
#         u^i_t choosen knowing xi_1 .. xi_t
#############################################################################

using StochDynamicProgramming, Clp
println("library loaded")

run_sddp = true # false if you don't want to run sddp
run_sdp  = false # false if you don't want to run sdp
test_simulation = false # false if you don't want to test your strategies

######## Optimization parameters  ########
# choose the LP solver used.
const SOLVER = ClpSolver() 			   # require "using Clp"
#const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0) # require "using CPLEX"

# convergence test
const MAX_ITER = 100 # number of iterations of SDDP

const step = 0.1   # discretization step of SDP

######## Stochastic Model  Parameters  ########
const N_STAGES = 6              # number of stages of the SP problem
const N_STOCKS = 3              # number of stocks of the SP problem
const COSTS = [sin(3*t)-1 for t in 1:N_STAGES]
#const COSTS = rand(N_STAGES)    # randomly generating deterministic costs


const CONTROL_MAX = 0.5         # bounds on the control
const CONTROL_MIN = 0

const XI_MAX = 0.3              # bounds on the noise
const XI_MIN = 0
const N_XI = 3                 # discretization of the noise

const r = 0.5                  # bound on cumulative control : \sum_{i=1}^N u_i < rN

const S0 = [0.5 for i=1:N_STOCKS]     # initial stock

# create law of noises
proba = 1/N_XI*ones(N_XI) # uniform probabilities
xi_support = collect(linspace(XI_MIN,XI_MAX,N_XI))
xi_law = StochDynamicProgramming.noiselaw_product([NoiseLaw(xi_support, proba) for i=1:N_STOCKS]...)
xi_laws = NoiseLaw[xi_law for t in 1:N_STAGES-1]

# Define dynamic of the stock:
function dynamic(t, x, u, xi)
    return [x[i] + u[i] - xi[i] for i in 1:N_STOCKS]
end


# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COSTS[t] *sum(u)
end

# constraint function
constraints_dp(t, x, u, w) = sum(u) <= r*N_STOCKS
constraints_sddp(t, x, u, w) = [sum(u) - r*N_STOCKS]

######## Setting up the SPmodel
s_bounds = [(0, 1) for i = 1:N_STOCKS]			# bounds on the state
u_bounds = [(CONTROL_MIN, CONTROL_MAX) for i = 1:N_STOCKS] # bounds on controls
spmodel = LinearSPModel(N_STAGES,u_bounds,S0,cost_t,dynamic,xi_laws, ineqconstr=constraints_sddp)
set_state_bounds(spmodel, s_bounds) 	# adding the bounds to the model
println("Model set up")

######### Solving the problem via SDDP
if run_sddp
    tic()
    println("Starting resolution by SDDP")
    # 10 forward pass, stop at MAX_ITER
    paramSDDP = SDDPparameters(SOLVER,
                               passnumber=1,
                               max_iterations=MAX_ITER)
    sddp = @time solve_SDDP(spmodel, paramSDDP, 2,  # display information every 2 iterations
                      stopcrit=IterLimit(MAX_ITER))
    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, sddp.bellmanfunctions)
    println("Lower bound obtained by SDDP: "*string(round(lb_sddp,4)))
    toc(); println();
end

######### Solving the problem via Dynamic Programming
if run_sdp
    tic()
    println("Starting resolution by SDP")
    stateSteps = [step for i=1:N_STOCKS] # discretization step of the state
    controlSteps = [step for i=1:N_STOCKS] # discretization step of the control
    infoStruct = "HD" # noise at time t is known before taking the decision at time t

    paramSDP = SDPparameters(spmodel, stateSteps, controlSteps, infoStruct)
    spmodel_sdp = StochDynamicProgramming.build_sdpmodel_from_spmodel(spmodel)
    spmodel_sdp.constraints = constraints_dp

    Vs = solve_dp(spmodel_sdp, paramSDP, 1)
    value_sdp = StochDynamicProgramming.get_bellman_value(spmodel,paramSDP,Vs)
    println("Value obtained by SDP: "*string(round(value_sdp,4)))
    toc(); println();
end

######### Comparing the solutions on simulated scenarios.
#srand(1234) # to fix the random seed accross runs
if run_sddp && run_sdp && test_simulation
    scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws,1000)
    costsddp, stocks = forward_simulations(spmodel, paramSDDP, sddp.solverinterface, scenarios)
    costsdp, states, controls = forward_simulations(spmodel,paramSDP, Vs, scenarios)
    println("Simulated relative gain of sddp over sdp: "
            *string(round(200*mean(costsdp-costsddp)/abs(mean(costsddp+costsdp)),3))*"%")
end

