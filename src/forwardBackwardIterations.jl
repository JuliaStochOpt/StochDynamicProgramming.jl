#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define the Forward / Backward iterations of the SDDP algorithm
#############################################################################

#=
"
Make a forward pass of the algorithm
Simulate a scenario of noise and compute an optimal trajectory on this
scenario according to the current value functions.
Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize
- param (SDDPparameters)
    the parameters of the SDDP algorithm
- V (bellmanFunctions)
    the current estimation of Bellman's functions 
- forwardPassNumber (int)
    number of forward simulation
    
- xi (Array{float}) 
    the noise scenarios on which we simulate, each line being one scenario. 
    Generated if not given.
- returnCosts (Bool)
    return the cost of each simulated scenario if true
- returnStocks (Bool)
    return the trajectory of the stocks if true
- returnControls (Bool)
    return the trajectory of controls if true
Returns (according to the last parameters):
- costs (Array{Float64,1})
    an array of the simulated costs
- stocks (Array{Float64,3})
    the simulated stock trajectories. stocks(k,t,:) is the stock for scenario k at time t.
- controls (Array{Float64,3})
    the simulated controls trajectories. controls(k,t,:) is the control for scenario k at time t.
"
=#

type Results
     co
    sto
    contr
end



function forward_simulations(model::LinearDynamicLinearCostSPmodel,
                            param::SDDPparameters,
                            V::Vector{PolyhedralFunction},
                            forwardPassNumber::Int64, 
                            xi,# = nothing,
                            returnCosts::Bool,  
                            returnStocks::Bool, 
                            returnControls::Bool)

    # TODO simplify if returnStocks=false 
    # TODO stock Controls

    # TODO declare stock as an array of states
    # specify initial state stocks[k,0]=x0
    # TODO generate scenarios xi
     costs = zeros(0)

     for k=1:param.forwardPassNumber
     if returnCosts  
          costs = zeros(k); 
     end
          for t=1:model.stageNumber-1 
               solution = solveOneStepOneAlea(    model,
                                                  param,
                                                  V,
                                                  t,
                                                  squeeze(stocks[k,t,:],1)'[:,1],
                                                  xi[:,t],
                                                  false, 
                                                  true,
                                                  true,
                                                  false)
               stocks[k,t+1,:] = solution[1:model.dimStates[t]];
               opt_control[k,t+1,:]= solution[model.dimStates[t]+1:model.dimStates[t]+model.dimControls[t]];
               #stocks[t+1] = convert(typeof(stocks[t]),sol[1:model.dimStates[t]])
               
               if returnCosts 
                 #costs[k] += costFunction(t,stocks[k,t,:],opt_control[k,t,:],xi[k,t]); #TODO
               end
         end
     end
     result = Results(costs,stocks,opt_control);
	
          
     return result; # adjust according to what is asked
end




#=
"
     add to Vt a cut of the form Vt >= beta + <lambda,.>
     Parameters:
     - Vt (bellmanFunction) 
         Current lower approximation of the Bellman function at time t
     - beta (Float) 
         affine part of the cut to add
     - lambda (Array{float,1})
         subgradient of the cut to add
"
=#






function addCut!(Vt::PolyhedralFunction, beta::Float64, lambda::Array{Float64,2})
     #TODO add >= beta + <lambda,.-x>,
     Vt.betas       = [Vt.betas;beta];
     Vt.lambdas     = [Vt.lambdas;lambda];
end


#=
"
Make a backward pass of the algorithm
For t:T-1 -> 0, compute a valid cut of the Bellman function
Vt at the state given by stocks and add them to 
the current estimation of Vt.
Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize
- param (SDDPparameters)
    the parameters of the SDDP algorithm
    
- V (bellmanFunctions)
    the current estimation of Bellman's functions
- stocks (Array{Float64,3})
    stocks[k,t,:] is the vector of stock where the cut is computed
    for scenario k and time t.
Return nothing

"
=#




function backward_pass(  model::LinearDynamicLinearCostSPmodel,
                         param::SDDPparameters,
                         V::Vector{PolyhedralFunction},
                         stocks,#::Array{float,3}
                         xi
                        )
                      
     for t=(model.stageNumber-1):-1:1
         for k =1:param.forwardPassNumber
             cost = zeros(1);
             subgradient = zeros(model.dimStates[t]);#TODO access
             #for w in 1:nXi[t] #TODO number of alea at t + can be parallelized
               solution = solveOneStepOneAlea(  model,
                                                  param,
                                                  V,
                                                  t,
                                                  squeeze(stocks[k,t,:],1)'[:,1],
                                                  xi[:,t],
                                                  false, 
                                                  false,
                                                  true,
                                                  true);
               cost = solution[end];#TODO
               subgradient = solution[1:(end-1)];#TODO  
               #cost+= prob[w,t]*costw;#TODO obtain probabilityz
               #subgradientw+=prob[w,t]*subgradientw;#TODO                      
             #end
             #beta = cost - subgradientw*stocks[k,t,:]#TODO dot product not working
             addCut!(V[t],cost,subgradient');
         end
     end
end
