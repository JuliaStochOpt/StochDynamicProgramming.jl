#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Test impact of risk solving a stock problem :
# Min   F [\sum_{t=1}^TF c_t u_t]
# s.t.    s_{t+1} = s_t + u_t - xi_t, s_0 given
#         0 <= s_t <= 1
#         u_min <= u_t <= u_max
#         u_t choosen knowing xi_1 .. xi_t
#############################################################################

using StochDynamicProgramming, Clp
println("library loaded")

run_Expectation   = true # false if you don't want to test Expectation
run_AVaR          = true # false if you don't want to test AVaR
run_WorstCase     = true # false if you don't want to test WorstCase
run_ConvexCombi   = true # false if you don't want to test Convex Combination
run_Polyhedral    = true # false if you don't want to test Polyhedral Risk Measure

######## Optimization parameters  ########
# choose the LP solver used.
SOLVER = ClpSolver() 			   # require "using Clp"
#const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0) # require "using CPLEX"

# convergence test
MAX_ITER = 10 # number of iterations of SDDP

######## Stochastic Model  Parameters  ########
N_STAGES = 6              # number of stages of the SP problem
COSTS = [sin(3*t)-1 for t in 1:N_STAGES-1]
#const COSTS = rand(N_STAGES)    # randomly generating deterministic costs

CONTROL_MAX = 0.5         # bounds on the control
CONTROL_MIN = 0

XI_MAX = 0.3              # bounds on the noise
XI_MIN = 0
N_XI = 10                 # discretization of the noise

S0 = 0.5                  # initial stock

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
s_bounds = [(0, 1)] 			# bounds on the state
u_bounds = [(CONTROL_MIN, CONTROL_MAX)] # bounds on controls

println("Initializing functions to compare execution time")
spmodel = LinearSPModel(N_STAGES,u_bounds,[S0],cost_t,dynamic,xi_laws, riskMeasure = Expectation())
set_state_bounds(spmodel, s_bounds) 	# adding the bounds to the model
# 10 forward pass, stop at MAX_ITER
paramSDDP = SDDPparameters(SOLVER,
                           passnumber=10,
                           max_iterations=MAX_ITER)
sddp = solve_SDDP(spmodel, paramSDDP, 0) # display information every 2 iterations
lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, sddp.bellmanfunctions)

######### Solving the problem via SDDP with Expectation
if run_Expectation
    tic()
    spmodel = LinearSPModel(N_STAGES,u_bounds,[S0],cost_t,dynamic,xi_laws, riskMeasure = Expectation())
    set_state_bounds(spmodel, s_bounds) 	# adding the bounds to the model
    println("Expectation's model set up")
    println("Starting resolution with Expectation")
    # 10 forward pass, stop at MAX_ITER
    paramSDDP = SDDPparameters(SOLVER,
                               passnumber=10,
                               max_iterations=MAX_ITER)
    sddp = solve_SDDP(spmodel, paramSDDP, 2) # display information every 2 iterations
    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, sddp.bellmanfunctions)
    println("Lower bound obtained by SDDP: "*string(round(lb_sddp,4)))
    toc(); println();
end

######### Solving the problem via SDDP with AVaR
if run_AVaR
    tic()
    spmodel = LinearSPModel(N_STAGES,u_bounds,[S0],cost_t,dynamic,xi_laws, riskMeasure = AVaR((N_XI-1)/N_XI))
    set_state_bounds(spmodel, s_bounds) 	# adding the bounds to the model
    println("AVaR's model set up")
    println("Starting resolution with AVaR")
    # 10 forward pass, stop at MAX_ITER
    paramSDDP = SDDPparameters(SOLVER,
                               passnumber=10,
                               max_iterations=MAX_ITER)
    sddp = solve_SDDP(spmodel, paramSDDP, 2) # display information every 2 iterations
    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, sddp.bellmanfunctions)
    println("Lower bound obtained by SDDP: "*string(round(lb_sddp,4)))
    toc(); println();
end

######### Solving the problem via SDDP with Worst Case
if run_WorstCase
    tic()
    spmodel = LinearSPModel(N_STAGES,u_bounds,[S0],cost_t,dynamic,xi_laws, riskMeasure = WorstCase())
    set_state_bounds(spmodel, s_bounds) 	# adding the bounds to the model
    println("Worst Case's model set up")
    println("Starting resolution with Worst Case")
    # 10 forward pass, stop at MAX_ITER
    paramSDDP = SDDPparameters(SOLVER,
                               passnumber=10,
                               max_iterations=MAX_ITER)
    sddp = solve_SDDP(spmodel, paramSDDP, 2) # display information every 2 iterations
    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, sddp.bellmanfunctions)
    println("Lower bound obtained by SDDP: "*string(round(lb_sddp,4)))
    toc(); println();
end

if run_ConvexCombi
    tic()
    spmodel = LinearSPModel(N_STAGES,u_bounds,[S0],cost_t,dynamic,xi_laws, riskMeasure = ConvexCombi((N_XI-1)/N_XI,0.5))
    set_state_bounds(spmodel, s_bounds) 	# adding the bounds to the model
    println("Convex Combination's model set up")
    println("Starting resolution with Convex Combination")
    # 10 forward pass, stop at MAX_ITER
    paramSDDP = SDDPparameters(SOLVER,
                               passnumber=10,
                               max_iterations=MAX_ITER)
    sddp = solve_SDDP(spmodel, paramSDDP, 2) # display information every 2 iterations
    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, sddp.bellmanfunctions)
    println("Lower bound obtained by SDDP: "*string(round(lb_sddp,4)))
    toc(); println();
end

if run_Polyhedral
    beta = (N_XI-1)/N_XI
    prob = 1/N_XI*ones(N_XI)
    polyset = repmat(1/beta*prob',N_XI)
    for i = 1:N_XI
        polyset[i,i] = (beta*N_XI-N_XI+1)/(N_XI*beta)
    end
    tic()
    spmodel = LinearSPModel(N_STAGES,u_bounds,[S0],cost_t,dynamic,xi_laws, riskMeasure = PolyhedralRisk(polyset))
    set_state_bounds(spmodel, s_bounds) 	# adding the bounds to the model
    println("Polyhedral's model set up")
    println("Starting resolution with Polyhedral")
    # 10 forward pass, stop at MAX_ITER
    paramSDDP = SDDPparameters(SOLVER,
                               passnumber=10,
                               max_iterations=MAX_ITER)
    sddp = solve_SDDP(spmodel, paramSDDP, 2) # display information every 2 iterations
    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, sddp.bellmanfunctions)
    println("Lower bound obtained by SDDP: "*string(round(lb_sddp,4)))
    toc(); println();
end
