#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Define the extensive formulation to check the results of small problems
#############################################################################


function extensive_formulation(model, 
                               param)


    const N_STAGES = 3
    const N_SCENARIOS = 2

    # FINAL TIME:
    const TF = N_STAGES

    # COST:
    const COST = -66*2.7*(1 + .5*(rand(TF) - .5))

    # Constants:
    const V_MAX = 100
    const V_MIN = 0

    const C_MAX = round(Int, .4/7. * VOLUME_MAX) + 1
    const C_MIN = 0
    
    const X_init = [50, 50]

    laws = model.noises
    T = model.stageNumber-1
    println(T)
    println(laws)
    
    mod = Model(solver=param.solver)
    
    N = Array{Int64,2}(zeros(T+1,1))

    N[1] = 1

    for t = 1 : (T)
        N[t+1] = N[t]*laws[t].supportSize # number of nodes at time t
    end
    
    #TODO what if there is more than two scenario at each steps
    #TODO use bounds
    @defVar(mod,  V_MIN <= x[t=1:T+1,n=1:N_SCENARIOS*N[t]] <= V_MAX)
    @defVar(mod,  C_MIN <= u[t=1:T,n=1:N_SCENARIOS*N[t+1]] <= C_MAX)
    @defVar(mod,  c[t=1:T,n=1:N_SCENARIOS*N[t]])
    
    proba = Any[]
    push!(proba, laws[1].proba)
    for t = 2 : T
       push!(proba, zeros(N[t+1]))
       for j = 1 : N[t]
          for k = 1:N_SCENARIOS
             proba[t][N_SCENARIOS*(j-1)+k] = laws[t].proba[k]*proba[t-1][j] 
             #proba[t][N_SCENARIOS*j] = laws[t].proba[2]*proba[t-1][j]
          end
       end
    end

    
    for t = 1 : (T)
        println("\n\nt=",t)
        for n = 1 : N[t]
            for xi = 1 : laws[t].supportSize
                m = (n-1)*laws[t].supportSize+xi
                @addConstraint(mod, [x[t+1,N_SCENARIOS*(m-1)+1];x[t+1,N_SCENARIOS*m]] .== 
                model.dynamics(t,[x[t,N_SCENARIOS*(n-1)+1];x[t,N_SCENARIOS*n]], [u[t,N_SCENARIOS*(m-1)+1];u[t,N_SCENARIOS*m]], laws[t].support[xi]))
                @addConstraint(mod, c[t,m] == model.costFunctions(t, [x[t,N_SCENARIOS*(n-1)+1];x[t,N_SCENARIOS*n]], [u[t,N_SCENARIOS*(m-1)+1];u[t,N_SCENARIOS*m]], laws[t].support[xi]))
            end
        end
    end
    
    #TODO use initial state and cost
    #@addConstraint(mod, c[1,1] == 0.)
    @addConstraint(mod, x[1,1] == X_init[1])
    @addConstraint(mod, x[1,2] == X_init[2])
    
    #TODO Calcul proba branche
    @setObjective(mod, Min, sum{proba[t][2*(n-1)+1]*c[t,2*(n-1)+1]+proba[t][2*n]*c[t,2*n] , t = 1:T, n=1:div(N[t+1],2)})
    
    status = solve(mod)
    
    solved = (string(status) == "Optimal")
    
    #println(mod)
    
    if solved
        println("resultat = ", getObjectiveValue(mod))
    end
end
