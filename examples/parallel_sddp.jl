
import StochDynamicProgramming

require("damsvalley.jl")

# Perform n parallel passes:
N_PARALLEL_COMPUTATIONS = 3
# Synchronize cuts every n iterations:
SYNCHRONIZE = 5

# Import model and params in main worker
# /!\ This line must be before redefinition of seed!
model, params = init_problem()

# Redefine seeds in every processes to maximize randomness:
@everywhere srand()

params.maxItNumber = SYNCHRONIZE
# First pass of algorithm to define value functions in memory:
V = StochDynamicProgramming.solve_SDDP(model, params)[1]

# Count number of available CPU:
ncpu = nprocs() - 1
println("\nLaunch simulation on ", ncpu, " processes")
workers = procs()[2:end]

# As we distribute computation in n process, we perform forward pass in parallel:
params.forwardPassNumber = max(1, round(Int, params.forwardPassNumber/ncpu))

# Start parallel computation:
for i in 1:N_PARALLEL_COMPUTATIONS
    # Distribute computation of SDDP in each process:
    refs = [@spawnat w StochDynamicProgramming.solve_SDDP(model, params, V, 1)[1] for w in workers]
    # Catch the result in the main process:
    V = StochDynamicProgramming.catcutsarray([fetch(r) for r in refs]...)
    # We clean the resultant cuts:
    StochDynamicProgramming.remove_redundant_cuts!(V)
    StochDynamicProgramming.prune_cuts!(model, params, V)
    println("Lower bound at pass ", i, ": ", StochDynamicProgramming.get_lower_bound(model, params, V))
end

