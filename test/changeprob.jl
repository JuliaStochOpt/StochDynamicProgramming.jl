using StochDynamicProgramming, JuMP, Clp, CutPruners
using Base.Test

EPSILON = 0.0001

@testset "Risk measure definition" begin
    @test isa(Expectation(), RiskMeasure)
    @test isa(WorstCase(), RiskMeasure)
    for i in 0:10
        @test isa(CVaR(i/10), RiskMeasure)
        for j in 0:10
          @test isa(ConvexCombi(i/10, j/10), RiskMeasure)
        end
    end
end

@testset "Equality WorstCase CVaR(1)" begin
    n = 100
    prob = 1/n*ones(n)
    @test sum(abs.(change_proba_risk(prob,CVaR(1),1:n)-change_proba_risk(prob,WorstCase(),1:n)) .<= EPSILON*ones(n)) == n
end

@testset "Equality Expectation CVaR(0)" begin
    n = 100
    prob = 1/n*ones(n)
    @test sum(abs.(change_proba_risk(prob,CVaR(0),1:n)-change_proba_risk(prob,Expectation(),1:n)) .<= EPSILON*ones(n)) == n
end

@testset "Equality Expectation ConvexCombi(beta,1)" begin
    n = 100
    prob = 1/n*ones(n)
    @test sum(abs.(change_proba_risk(prob,Expectation(),1:n)-change_proba_risk(prob,ConvexCombi(rand(),1),1:n)) .<= EPSILON*ones(n)) == n
end

@testset "Equality CVaR(beta) ConvexCombi(beta,0)" begin
    n = 100
    beta = rand()
    prob = 1/n*ones(n)
    @test sum(abs.(change_proba_risk(prob,CVaR(beta),1:n)-change_proba_risk(prob,ConvexCombi(beta,0),1:n)) .<= EPSILON*ones(n)) == n
end

@testset "Right sense of CVaR" begin
    betamin = rand()/2
    betamax = rand()/2+0.5
    n = 100
    prob = 1/n*ones(n)
    @test change_proba_risk(prob, CVaR(betamax), 1:n)'*(n:-1:1) - change_proba_risk(prob, CVaR(betamin), 1:n)'*(n:-1:1) >= 0
end
