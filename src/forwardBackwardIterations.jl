#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define the Forward / Backward iterations of the SDDP algorithm
#############################################################################



function forwardSimulations(n::int, 
                            xi = nothing,
                            returnCosts::Bool = true,  
                            returnStocks::Bool=true, 
                            returnControls::Bool = false)
# Simulate n trajectories given by the value functions V

# TODO simplify if returnStocks=false 
# TODO stock Controls

# TODO declare stock as an array of states
# specify initial state stocks[k,0]=x0
# TODO generate scenarios xi
if returnCosts  
    costs = zeros(k); 
end

for k = 1:n #TODO can be parallelized + some can be dropped if too long
    
    for t=0:T-1 #TODO get T
        stocks[k,t+1], opt_control = solveOneStepOneAlea(t,stocks[k,t],xi[k,t],
                                    returnOptNextStep=true, 
                                    returnOptControl=true,
                                    returnSubgradient=false,
                                    returnCost=false);
        if returnCosts 
            costs[k] += costFunction(t,stocks[k,t],opt_control,xi[k,t]); #TODO
        end
    end
end
return costs,stocks # adjust according to what is asked
end

function addCut(t,x, beta, lambda)
    #TODO add >= beta + <lambda,.-x>,
end

function backwardPass(stockTrajectories)
for t=T-1:0
    for k in #TODO
        cost = zeros(1);
        subgradient = zeros(dimStates[t]);#TODO access
        for w in 1:nXi[t] #TODO + can be parallelized
            subgradientw, costw = solveOneStepOneAlea(t,stockTrajectories[k,t],xi[t,],
                                        returnOptNextStep=false, 
                                        returnOptControl=false,
                                        returnSubgradient=true,
                                        returnCost=true);
            cost+= prob[w,t]*costw;#TODO
            subgradientw+=prob[w,t]*subgradientw;#TODO                      
        end
    addCut(t,stockTrajectories[k,t],subgradient,cost);
    end
end
end
