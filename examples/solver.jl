
#= using Clp =#
#= OPTIMIZER = optimizer_with_attributes(Clp.Optimizer, "LogLevel"=>0) =#
using Xpress
OPTIMIZER = optimizer_with_attributes(Xpress.Optimizer, "OUTPUTLOG"=>0)
#= using Gurobi =#
#SOLVER = Gurobi.GurobiSolver(OutputFlag=false, Threads=1)
#= OPTIMIZER = optimizer_with_attributes(Gurobi.Optimizer, =#
#=     "OutputFlag"=>0) =#
#= using CPLEX =#
#= SOLVER = CPLEX.CplexSolver(CPX_PARAM_SIMDISPLAY=0, CPX_PARAM_BARDISPLAY=0) =#
