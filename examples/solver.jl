

#= using Clp =#
#= SOLVER = ClpSolver() =#
using Gurobi
SOLVER = Gurobi.GurobiSolver(OutputFlag=false, Threads=1)
#= using CPLEX =#
#= SOLVER = CPLEX.CplexSolver(CPX_PARAM_SIMDISPLAY=0, CPX_PARAM_BARDISPLAY=0) =#
