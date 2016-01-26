#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Model and solve the One-Step One Alea problem in different settings
# - used to compute the optimal control (in forward phase / simulation)
# - used to compute the cuts in the Backward phase
#############################################################################

#=
"
Solve the Bellman equation at time t starting at state x under alea xi
with the current evaluation of Vt+1
The function solve
min_u current_cost(t,x,u,xi) + current_Bellman_Value_{t+1}(dynamic(t,x,u,xi))
and can return the optimal control and a subgradient of the value of the 
problem with respect to the initial state x
Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize
- param (SDDPparameters)
    the parameters of the SDDP algorithm
    
- V (bellmanFunctions)
    the current estimation of Bellman's functions
    
- t (int)
    time step at which the problem is solved
- x (Array{Float})
    current starting state 
    
- xi (Array{float}) 
    current noise value
- returnOptNextStage (Bool)
    return the optimal state at t+1
- returnOptcontrol (Bool)
    return the optimal control
- returnSubgradient (Bool)
    return the subgradient
- returnCost (Bool)
    return the value of the problem
Returns (according to the last parameters):
- costs (Array{float,1})
    an array of the simulated costs
- stocks (Array{float})
    the simulated stock trajectories. stocks(k,t,:) is the stock for scenario k at time t.
- controls (Array{float})
    the simulated controls trajectories. controls(k,t,:) is the control for scenario k at time t.
"
=#

Cxi 		= [1 4 5; 1 4 5];
#TODO include the random case

function solveOneStepOneAlea(model::LinearDynamicLinearCostSPmodel,
                            param::SDDPparameters,
                            V::Vector{PolyhedralFunction},
                            t,
                            x::Vector{Float64},
                            xi::Vector{Float64},
                            returnOptNextStage::Bool=false, 
                            returnOptControl::Bool=false,
                            returnSubgradient::Bool=false,
                            returnCost::Bool=false)
    
    	#TODO call the right following function
    	# return (optNextStep, optControl, subgradient, cost) #depending on which is asked

	#auxiliary variable
	lengthx  = model.dimStates[t];
	lengthu  = model.dimControls[t];
	lengthxi = length(Cxi[:,t]);

	Lambdas = V[t+1].lambdas;

	lengthV  = length(Lambdas[:,1]);

	#Cx is the cost matrix of the state x
	#Cu is the cost matrix of the control u
	#Cw is the cost matrix of the noise w
	#cost 	= [Cx[:,t];Cu[:,t];zeros(lengthx,1);1.0];
	cost = model.costFunctions(t,x,x,xi);
	

	#Vlambdas and Vbetas let us define the cutting hyperplanes, Vlambdas containing the direction of the hyperplanes and Vbetas the origin intercep
	#Lambdas = Vlambdas[:,t+1]; 
	#Betas 	= Vbetas[:,t+1];

	Lambdas = V[t+1].lambdas;
	Betas	 = V[t+1].betas;

	#multiplier
	sensemul = ['=' for i in 1:(lengthx)];
	Amul 	 = [eye(lengthx) zeros(lengthx,lengthu+lengthx+1)];
	bmul	 = collect(x);

	#dynamic
	sensedyn = ['=' for i in 1:(lengthxi)];
	#Adyn	 = dynamique[t]';
	Adyn 	 = model.dynamics(t,x,x,xi); 	
	bdyn	 = (-1.0).*xi;

	#The inequality constraint are the le linearisation of 'theta = max_of_cuts'
	Acut 	= [-Lambdas zeros(lengthV,lengthu+lengthx) ones(lengthV,1)];
	bcut 	= -Lambdas*(squeeze(stocks[1,t+1,:],1)'[:,1])+Betas;
	sensecut= ['>' for i in 1:(lengthV)];

	#We define aggregate matrix to give the problem to a solver
	#TODO Define properly the matric A[t]
	#A 	= [A[t];Acut];
	#b 	= [b[t];bcut];
	#sense 	= [sense[t] ; sensecut];
	
	A 	= [Amul;Adyn;Acut];
	b 	= [bmul;bdyn;bcut];
	sense 	= [sensemul;sensedyn;sensecut];	

	#Without other constraints, the lower bounds are set to zero and the upper bounds are set to infinite
	LB	= [zeros(lengthx+lengthu+lengthx,1);-Inf];
	UB	= Inf*ones(length(cost),1);

	#An external linear solver is called here
	solution=linprog(cost[1:length(cost)], A, sense[1:length(sense)], b[1:length(b)], LB[1:length(LB)], UB[1:length(UB)], param.solver);
	
	#TODO retourner obj + le prix de l'aléa
	result = Float64[]	
	if (returnOptNextStage)
		result = [result;solution.sol[1:lengthx]];#TODO correct this formula
	end
        if (returnOptControl)
		result = [result; solution.sol[lengthx+1:lengthx+lengthu]];
	end
	if (returnSubgradient)
		result = [result; solution.attrs[:lambda][1:(1+lengthx-1)]];
	end
        if (returnCost)
		result = [result;solution.objval+Cxi[:,t]'*xi];
	end
	
	result;
end
