#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Model and solve the One-Step One Alea problem in different settings
# - used to compute the optimal control (in forward phase / simulation)
# - used to compute the cuts in the Backward phase
#############################################################################


#TODO include the random case



using MathProgBase #TODO interface with JuMP
using GLPKMathProgInterface


#T=3

#Cx 		= [3 4 2; 3 4 2];
#Cu 		= [2 1 1; 2 1 1];
Cxi 		= [1 4 5; 1 4 5];
#X		= [5 3 3; 5 3 3];
#Xi 		= [1 2 3; 1 2 3];
#Vbetas		= [7 25 6; 7 25 6];
#Vlambdas	= [7 5 2; 7 5 2];
#time 		= 2;

#A1        	= [1.0 0 -1.0 0 -1.0 0 0.0 ; 0 1.0 0 -1.0 -1.0 0 0.0];
#A2        	= [1.0 0 -1.0 0 -1.0 0 0.0 ; 0 1.0 0 -1.0 0 -1.0 0.0];
#A3 	  	= []; #TODO integrate the fact that there is no dynamic for the final step
#dynamique 	= Any[A1',A2'];

#function solveOneStepOneAlea(t::int,
#                                    x,
#                                    xi,
#                                    returnOptNextStep::Bool=false, 
#                                    returnOptControl::Bool=false,
#                                    returnSubgradient::Bool=false,
#                                    returnCost::Bool=false)
#    
#    #TODO call the right following function
#    # return (optNextStep, optControl, subgradient, cost) #depending on which is asked
#end

function solveOneStepOneAleaLinear(t,  #TODO type
                                    x, #TODO type
                                    xi,#TODO type
                                    returnOptNextStep::Bool=false, 
                                    returnOptControl::Bool=false,
                                    returnSubgradient::Bool=false,
                                    returnCost::Bool=false)
    #TODO call the solver on the linear problem
    #TODO return optNextStep, optControl, optValue, subgradient 

	#auxiliary variable
	lengthx  = test.dimStates[t];
	lengthu  = test.dimControls[t];
	lengthxi = length(Cxi[:,t]);

	Lambdas = cut[t+1].lambdas;

	lengthV  = length(Lambdas[:,1]);

	#Cx is the cost matrix of the state x
	#Cu is the cost matrix of the control u
	#Cw is the cost matrix of the noise w
	#cost 	= [Cx[:,t];Cu[:,t];zeros(lengthx,1);1.0];
	cost = test.costFunctions(t,x,x,xi);
	

	#Vlambdas and Vbetas let us define the cutting hyperplanes, Vlambdas containing the direction of the hyperplanes and Vbetas the origin intercep
	#Lambdas = Vlambdas[:,t+1]; 
	#Betas 	= Vbetas[:,t+1];

	Lambdas = cut[t+1].lambdas;
	Betas	 = cut[t+1].betas;

	#multiplier
	sensemul = ['=' for i in 1:(lengthx)];
	Amul 	 = [eye(lengthx) zeros(lengthx,lengthu+lengthx+1)];
	bmul	 = collect(x);

	#dynamic
	sensedyn = ['=' for i in 1:(lengthxi)];
	#Adyn	 = dynamique[t]';
	Adyn 	 = test.dynamics(t,x,x,xi); 	
	bdyn	 = (-1.0).*xi;

	#The inequality constraint are the le linearisation of 'theta = max_of_cuts'
	Acut 	= [-Lambdas zeros(lengthV,lengthu+lengthx) ones(lengthV,1)];
	bcut 	= -Lambdas*stockTrajectories[t+1]+Betas;
	sensecut= ['>' for i in 1:(lengthV)];

	#We define aggregate matrix to give the problem to a solver
	#TODO Define properly the matric A[t]
	#A 	= [A[t];Acut];
	#b 	= [b[t];bcut];
	#sense 	= [sense[t] ; sensecut];
	
	A 	= [Amul;Adyn;Acut];
	b 	= [bmul;bdyn;bcut];
	sense 	= [sensemul;sensedyn;sensecut];	

	#Without other constraints, the bouds are set to infinite
	LB	= [zeros(lengthx+lengthu+lengthx,1);-Inf];
	UB	= Inf*ones(length(cost),1);

	#An external linear solver is called here
	solution=linprog(cost[1:length(cost)], A, sense[1:length(sense)], b[1:length(b)], LB[1:length(LB)], UB[1:length(UB)], GLPKSolverLP(method=:Exact, presolve=true, msg_lev=GLPK.MSG_ON));

	#Coordinates of the new plane
	#lambda	=solution.attrs[:lambda][1:lengthx,1]
	#beta  	=solution.objval;
	
	#TODO retourner obj + le prix de l'aléa
	#[solution;lambda;beta]
	result = []	
	if (returnOptNextStep)
		result = [result;solution.objval+Cxi[:,t]'*xi];
	end
        if (returnOptControl)
		result = [result; solution.sol]
	end
	if (returnSubgradient)
		result = [result; solution.attrs[:lambda][1:(1+lengthx-1)]];
	end
        if (returnCost)
		result = [result; solution.attrs[:redcost]];
	end
	
	result;
end
