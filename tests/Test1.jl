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

println("beginning of instanciation")

alea = zeros(2,3);

test = LinearDynamicLinearCostSPmodel(3,[2;2;2],[2;2;2],[1;1],costFunction,dynamic,[4 4;4 4;4 4],[1000 1000;1000 1000;1000 1000],alea);

cut1 = PolyhedralFunction([1;1],[1 0;0 1]);
cut2 = PolyhedralFunction([2;2],[2 0;0 2]);
cut3 = PolyhedralFunction([3;3],[3 0;0 3]);

k=1;

cut = PolyhedralFunction[cut1,cut2,cut3];

#stockTrajectories = Any[[1.0;1.0],[2.0;2.0],[3.0;3.0]];
#stockTrajectories = ones(1,3,2);
stocks = zeros(k,test.stageNumber,2);

for t=1:test.stageNumber
     for i=1:2
          stocks[k,t,i] = test.lowerbounds[t,i]; 
     end;
end;

opt_control = zeros(1,3,2)





function initialization( model::LinearDynamicLinearCostSPmodel,
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
             #beta = cost - subgradientw*stocks[k,t,:] #TODO dot product not working
             V[t].betas       = [cost];
             V[t].lambdas     = subgradient';
         end
     end
end

partest = SDDPparameters(GLPKSolverLP(),1,initialization,[0;20]);

param.initialization(test,
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
