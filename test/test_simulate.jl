include("../src/simulate.jl")

supp = [1, 2, 3] 
p = [0.1 0.4 0.5] 
w1 = NoiseLaw_const(3,supp,p)
w2 = NoiseLaw(supp,p)
simulate([w1,w2],3)
