#############################################################################
# Test SDDP upon damsvalley with quadratic final cost
#############################################################################

##################################################
# Set a seed for reproductability:
srand(2713)

using StochDynamicProgramming, JuMP, CPLEX
##################################################


##################################################
# PROBLEM DEFINITION
##################################################
# We consider here a valley with 5 dams:
const N_DAMS = 5

const N_STAGES = 12
const N_ALEAS = 10

# Cost are negative as we sell the electricity produced by
# dams (and we want to minimize our problem)
const COST = -66*2.7*(1 + .5*(rand(N_STAGES) - .5))

# Constants:
const VOLUME_MAX = 80
const VOLUME_MIN = 0

const CONTROL_MAX = 40
const CONTROL_MIN = 0

# Define initial status of stocks:
const X0 = [40 for i in 1:N_DAMS]

# Dynamic of stocks:
const A = eye(N_DAMS)
# The problem has the following structure:
# dam1 -> dam2 -> dam3 -> dam4 -> dam5
# We need to define the corresponding dynamic:
const B =  [-1  0.  0.  0.  0.  -1  0.  0.  0.  0.;
            1.  -1  0.  0.  0.  1.  -1  0.  0.  0.;
            0.  1.  -1  0.  0.  0.  1.  -1  0.  0.;
            0.  0.  1.  -1  0.  0.  0.  1.  -1  0.;
            0.  0.  0.  1.  -1  0.  0.  0.  1.  -1]
# Define dynamic of the dam:
function dynamic(t, x, u, w)
    return A*x + B*u + w
end

# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COST[t] * sum(u[1:N_DAMS])
end

# We define here final cost a quadratic problem
# we penalize the final costs if it is greater than 40.
function final_cost_dams(model, m)
    # Here, model is the optimization problem at time T - 1
    # so that xf (x future) is the final stock
    alpha = JuMP.getvariable(m, :alpha)
    w = JuMP.getvariable(m, :w)
    x = JuMP.getvariable(m, :x)
    u = JuMP.getvariable(m, :u)
    xf = JuMP.getvariable(m, :xf)
    @JuMP.variable(m, z1 >= 0)
    @JuMP.variable(m, z2 >= 0)
    @JuMP.variable(m, z3 >= 0)
    @JuMP.variable(m, z4 >= 0)
    @JuMP.variable(m, z5 >= 0)
    @JuMP.constraint(m, alpha == 0.)
    @JuMP.constraint(m, z1 >= 40 - xf[1])
    @JuMP.constraint(m, z2 >= 40 - xf[2])
    @JuMP.constraint(m, z3 >= 40 - xf[3])
    @JuMP.constraint(m, z4 >= 40 - xf[3])
    @JuMP.constraint(m, z5 >= 40 - xf[3])
    @JuMP.objective(m, Min, model.costFunctions(model.stageNumber-1, x, u, w) + 500.*(z1*z1+z2*z2+z3*z3+z4*z4+z5*z5))
end

##################################################
# SDDP parameters:
##################################################
# Number of forward pass:
const FORWARD_PASS = 10.
const EPSILON = .05
# Maximum number of iterations
const MAX_ITER = 50
##################################################

"""Build probability distribution at each timestep.
Return a Vector{NoiseLaw}"""
function generate_probability_laws()
    laws = Vector{NoiseLaw}(N_STAGES-1)
    # uniform probabilities:
    proba = 1/N_ALEAS*ones(N_ALEAS)

    for t=1:N_STAGES-1
        support = rand(0:9, N_DAMS, N_ALEAS)
        laws[t] = NoiseLaw(support, proba)
    end
    return laws
end

"""Instantiate the problem."""
function init_problem()
    aleas = generate_probability_laws()

    x_bounds = [(VOLUME_MIN, VOLUME_MAX) for i in 1:N_DAMS]
    u_bounds = vcat([(CONTROL_MIN, CONTROL_MAX) for i in 1:N_DAMS], [(0., 200) for i in 1:N_DAMS]);
    model = LinearDynamicLinearCostSPmodel(N_STAGES,
                                                u_bounds,
                                                X0,
                                                cost_t,
                                                dynamic,
                                                aleas,
                                                final_cost_dams)

    # Add bounds for stocks:
    set_state_bounds(model, x_bounds)

    # We need to use CPLEX to solve QP at final stages:
    solver = CPLEX.CplexSolver(CPX_PARAM_SIMDISPLAY=0, CPX_PARAM_BARDISPLAY=0)
    params = SDDPparameters(solver, FORWARD_PASS, EPSILON, MAX_ITER)

    return model, params
end

# Solve the problem:
model, params = init_problem()
V, pbs = solve_SDDP(model, params, 1)

