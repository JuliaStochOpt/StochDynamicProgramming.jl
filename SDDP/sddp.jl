using MathProgBase;
using GLPKMathProgInterface;

function CutPerAlea(t,xt,w,Vlambda, Vbeta)
	#Cx is the cost matrix of the state x
	#Cu is the cost matrix of the control u
	#Cw is the cost matrix of the noise w
	cout 	= [Cx[:,t];Cu[:,t];Cw[:,t];1.0];

	#Vlambdas and Vbetas let us define the cutting hyperplanes, Vlambdas containing the direction of the hyperplanes.
	Lambdas = Vlambdas[:,t]; 
	Betas 	= Vbetas[:,t];

	#auxiliary variable
	taillex = length(Cx[:,t]);
	tailleu = length(Cu[:,t]);
	taillew = length(Cw[:,t]);
	tailleV = length(Vlambdas[:,t]);

	#The dual multiplier of the equality constraint 'x=xt' gives us the direction of the new hyperplane
	Aeq 	= [eye(taillex) zeros(taillex,tailleu+taillew+1)];
	beq 	= xt;

	#The inequality constraint are the le linearisation of 'theta = max_of_cuts'
	Ain 	= [zeros(tailleV,taillex+tailleu+taillew) ones(tailleV,1)];
	bin 	= Lambdas .* ft(xt,u,w) + Betas;

	#We define aggregate matrix to give the problem to a solver
	A 	=[Aeq;Ain];
	b 	=[beq;bin];
	sense 	= [['=' for i in 1:taillex] ; ['>' for i in 1:tailleV]];

	#Without other constraints, the bouds are set to infinite
	LB	=-Inf*ones(length(cout),1);
	UB	=Inf*ones(length(cout),1);

	#An external linear solver is called here
	solution=linprog(cout, A, sense, b, GLPKSolverLP(method=:Exact, presolve=true, msg_lev=GLPK.MSG_ON));

	#Coordinates of the new plane
	lambda	=solution.attrs[:lambda][1:taillex,1]
	beta  	=solution.objval;

end



