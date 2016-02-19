#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Define the extensive formulation to check the results of small problems
#############################################################################


function extensive_formulation(model, 
                               param,
                               COST)


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
    @defVar(mod,  0 <= x[t=1:T+1,n=1:2*N[t]] <= 100)
    @defVar(mod,  0 <= u[t=1:T,n=1:2*N[t+1]] <= 7)
    @defVar(mod,  c[t=1:T,n=1:2*N[t]])
    
    proba = Any[]
    push!(proba, laws[1].proba)
    for t = 2 : T
       push!(proba, zeros(N[t+1]))
       for j = 1 : N[t]
           proba[t][2*(j-1)+1] = laws[t].proba[1]*proba[t-1][j] 
           proba[t][2*j] = laws[t].proba[2]*proba[t-1][j]
       end
    end
    println("proba \n",proba)
    
    for t = 1 : (T)
        println("\n\nt=",t)
        for n = 1 : N[t]
            for xi = 1 : laws[t].supportSize
                m = (n-1)*laws[t].supportSize+xi
                @addConstraint(mod, [x[t+1,2*(m-1)+1];x[t+1,2*m]] .== model.dynamics(t,[x[t,2*(n-1)+1];x[t,2*n]], [u[t,2*(m-1)+1];u[t,2*m]], laws[t].support[xi]))
                @addConstraint(mod, c[t,m] == model.costFunctions(t, [x[t,2*(n-1)+1];x[t,2*n]], [u[t,2*(m-1)+1];u[t,2*m]], laws[t].support[xi]))
            end
        end
    end
    
    #TODO use initial state and cost
    #@addConstraint(mod, c[1,1] == 0.)
    @addConstraint(mod, x[1,1] == 50.)
    @addConstraint(mod, x[1,2] == 50.)
    
    #TODO Calcul proba branche
    @setObjective(mod, Min, sum{proba[t][2*(n-1)+1]*c[t,2*(n-1)+1]+proba[t][2*n]*c[t,2*n] , t = 1:T, n=1:div(N[t+1],2)})
    
    println(mod)
    println(laws)
    
    status = solve(mod)
    
    solved = (string(status) == "Optimal")
    
    if solved
        println("resultat = ",getObjectiveValue(mod))
        println("control = ",getValue(u))
    end
end
