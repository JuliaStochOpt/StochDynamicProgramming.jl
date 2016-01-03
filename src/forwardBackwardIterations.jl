#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define the Forward / Backward iterations of the SDDP algorithm
#############################################################################



function forwardSimulations(n)
# Simulate n trajectories given by the value functions V

# TODO simplify if returnStocks=false 
# TODO stock Controls

# TODO declare stock as an array of states
# specify initial state stocks[k,0]=x0
# TODO generate scenarios xi
#if returnCosts  
#    costs = zeros(k); 
#end

#for k = 1:n #TODO can be parallelized + some can be dropped if too long
    
    for t=1:(test.stageNumber-1) #TODO get T
        sol = solveOneStepOneAleaLinear(t,stockTrajectories[t],zeros(test.dimStates[t],1),
                                    false, 
                                    true,
                                    false,
                                    false);
	
	#stockTrajectories[t+1]=sol[1:test.dimStates[t]];
	stockTrajectories[t+1] = convert(typeof(stockTrajectories[t]),sol[1:test.dimStates[t]])
	#opt_control

        #if returnCosts 
        #    costs[k] += costFunction(t,stocks[k,t],opt_control,xi[k,t]); #TODO
        #end
    end
#end
#return costs,stocks # adjust according to what is asked
end

function addCut(beta, lambda, polyfunction)
    #TODO add >= beta + <lambda,.-x>,
	polyfunction.betas = [polyfunction.betas;beta];
	polyfunction.lambdas = [polyfunction.lambdas;lambda];
end

function backwardPass(stockTrajectories)
for t=(test.stageNumber-1):-1:1
    #for k = 1:1#TODO
        cost = zeros(1);
        subgradient = zeros(test.dimStates[t]);#TODO access
        sol = solveOneStepOneAleaLinear(t,stockTrajectories[t],zeros(test.dimStates[t],1),
                                        true, 
                                        false,
                                        true,
                                        false);
            cost = sol[1];#TODO
            subgradient = sol[2:end];#TODO                      
    addCut(cost,subgradient',cut[t]);
    #end
end
end
