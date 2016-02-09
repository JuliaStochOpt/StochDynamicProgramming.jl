#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  the actual optimization function 
#
#############################################################################


include("../src/SDDP.jl")
println("SDDP.jl file included");

println(" ");
println("All files well included");
println(" ");

println("beginning of instanciation");


Time = 3::Int64;
X = 2::Int64;

function costFunction(t,x,u,xi)
    #TODO
    	#Cx 	= [3 4 2; 3 4 2];
	#Cu 	= [2 1 1; 2 1 1];
	#Cxi 	= [1 4 5; 1 4 5];

     Cx 	= 10*rand(X,Time);
	Cu 	= 10*rand(X,Time);
	Cxi 	= 10*rand(X,Time);    
     
	lengthx  = length(Cx[:,t]);
	lengthu  = length(Cu[:,t]);
	lengthxi = length(Cxi[:,t]);

	cost 	= [Cx[:,t];Cu[:,t];Cxi[:,t];1.0];

	return cost
end

function dynamic(t,x,u,xi)
    #TODO
    A1 = [1.0 0 -1.0 0 -1.0 0 0.0 ; 0 1.0 0 -1.0 -1.0 0 0.0]
    dynamique = Any[A1];
    for i=2:Time-1
          A = [1.0 0 -1.0 0 -1.0 0 0.0 ; 0 1.0 0 -1.0 -1.0 0 0.0]
          dynamique = push!(dynamique,A);
    end
#	A1        = [1.0 0 -1.0 0 -1.0 0 0.0 ; 0 1.0 0 -1.0 -1.0 0 0.0];
#	A2        = [1.0 0 -1.0 0 -1.0 0 0.0 ; 0 1.0 0 -1.0 0 -1.0 0.0];
#	A3 	  = []; #TODO integrate the fact that there is no dynamic for the final step
#	A1        = [1.0 0 -1.0 0 -1.0 0 0.0 ; 0 1.0 0 -1.0 -1.0 0 0.0];
#	A2        = [1.0 0 -1.0 0 -1.0 0 0.0 ; 0 1.0 0 -1.0 0 -1.0 0.0];
#	A3 	  = [];
#	dynamique = Any[A1',A2'];

     dynamique = push!(dynamique,[]);     
     
	new_state = dynamique[t];
    	
	return new_state
end

alea = rand(2,3);

test = LinearDynamicLinearCostSPmodel(Time,X*ones(Time),X*ones(Time),10*rand(X,1),costFunction,dynamic,zeros(Time,X),1000*ones(Time,X),alea);

cut1 = PolyhedralFunction(10*rand(X),10*rand(X,X));
cut = PolyhedralFunction[cut1];
for i = 2 : Time
     cut2 = PolyhedralFunction(10*rand(X),10*rand(X,X));
     cut = push!(cut,cut2);
end


k=2;

#cut = PolyhedralFunction[cut1,cut2,cut3];

#stockTrajectories = Any[[1.0;1.0],[2.0;2.0],[3.0;3.0]];
#stockTrajectories = ones(1,3,2);
stocks = zeros(k,test.stageNumber,X);

for t=1:test.stageNumber
     for i=1:X
          stocks[k,t,i] = test.lowerbounds[t,i]; 
     end;
end;

opt_control = zeros(k,test.stageNumber,X)

prob = rand(Time);
prob = prob /(sum(prob));
bruit     = NoiseLaw(test.stageNumber,3*rand(X,Time),prob);
omeg = [bruit]
for i = 2 : Time
     bruitbis =  NoiseLaw(test.stageNumber,3*rand(X,Time),prob);
     omeg = push!(omeg,bruitbis);   
end
#bruitbis  = NoiseLaw(test.stageNumber,[4 5 6;4 5 6],[1/3;1/3;1/3])
#bruitter  = NoiseLaw(test.stageNumber,[7 8 9;7 8 9],[1/3;1/3;1/3])
#omeg = [bruit,bruitbis,bruitter]



function initialization( model::LinearDynamicLinearCostSPmodel,
                         param::SDDPparameters,
                         V::Vector{PolyhedralFunction},
                         stocks,#::Array{float,3}
                         xi
                        )
     
     for t=(model.stageNumber-1):-1:1
         for k =1:param.forwardPassNumber
             costw = zeros(model.stageNumber);
             subgradientw = zeros(omeg[t].supportSize,convert(Int64,model.dimStates[t]));#TODO access
             for w = 1:omeg[t].supportSize#nXi[t] #TODO number of alea at t + can be parallelized
               solution = solveOneStepOneAlea(  model,
                                                  param,
                                                  V,
                                                  t,
                                                  squeeze(stocks[k,t,:],1)'[:,1],
                                                  omeg[t].support[:,w],
                                                  false, 
                                                  false,
                                                  true,
                                                  true);
               costw[w] = solution[end];#TODO
               subgradientw[w,:] = solution[1:(end-1)];#TODO                        
             end
             cost = (omeg[t].proba)'*costw;#TODO obtain probabilityz
             subgradient = (omeg[t].proba)'*subgradientw;#TODO
             beta = cost - subgradient*squeeze(stocks[k,t,:],1)'[:,1] #TODO dot product not working
             V[t].betas       = beta;
             V[t].lambdas     = subgradient;
         end
     end
end

partest = SDDPparameters(GLPKSolverLP(),1,initialization,[0;20]);

partest.initialization(test,
               partest,
               cut,
               stocks,
               alea);
              
              
              
println("end of instanciation")

println(" ");
println("Launch of function optimize");
optimize(test,partest,cut,alea);

println("                _    _")
println("               (_)  | |");
println(" _____  _   _   _   | |");
println("|  _  || | | | | |  | |");
println("| |_| || |_| | | |  \\ /");
println("|_____||_____| |_|   o ");
