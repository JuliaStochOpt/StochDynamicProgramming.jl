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

using StochDynamicProgramming, JuMP, Clp, Distributions, Gurobi#CPLEX
println("library loaded")

run_sddp = true

######## Optimization parameters  ########
# choose the LP solver used.
#const SOLVER = ClpSolver()
#const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0)
#OPTIMIZER = Gurobi.Optimizer
OPTIMIZER = optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0) #Clp.Optimizer

# convergence test
const MAX_ITER = 1 # maximum iteration of SDDP

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
xi_support = collect(range(XI_MIN,stop=XI_MAX,length=N_XI))
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
spmodel = LinearSPModel(N_STAGES, u_bounds, [S0], cost_t, dynamic, xi_laws)
set_state_bounds(spmodel, s_bounds)


######### Define scenarios
#scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws, 1000)
scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws, 2)

######## Define different scenarios
paramSDDP1 = SDDPparameters(OPTIMIZER, passnumber=4, gap=0., max_iterations=MAX_ITER) #forwardpassnumber, sensibility
paramSDDP2 = SDDPparameters(OPTIMIZER, passnumber=10, gap=0., max_iterations=2)

######## Define parameters collection
paramSDDP = [paramSDDP1 for i in 1:4]

#Benchmark the collection of parameters
benchmark_parameters(spmodel, paramSDDP, scenarios, 12,verbosity=1)
