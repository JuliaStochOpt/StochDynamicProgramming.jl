using StochDynamicProgramming, JuMP, Clp
using Base.Test

EPSILON = 0.0001

# Check that there is no problem of definition
@testset "Risk measure definition" begin
    @test isa(Expectation(), RiskMeasure)
    @test isa(WorstCase(), RiskMeasure)
    @test isa(AVaR(0), RiskMeasure)
    @test isa(AVaR(0.5), RiskMeasure)
    @test isa(AVaR(1), RiskMeasure)
    @test isa(ConvexCombi(0,0), RiskMeasure)
    @test isa(ConvexCombi(0,1), RiskMeasure)
    @test isa(ConvexCombi(1,0), RiskMeasure)
    @test isa(ConvexCombi(1,1), RiskMeasure)
    @test isa(ConvexCombi(0.5,0.5), RiskMeasure)
end

# Test limit cases
# An AVaR with beta = 0 is a WorstCase
# An AVaR with beta = 1 is an Expectation
# A ConvexCombi with lambda = 0 is an AVaR
# A ConvexCombi with lambda = 1 is an Expectation
@testset "Equality formulations" begin
    @testset "Equality WorstCase AVaR(0)" begin
        n = 100
        prob = 1/n*ones(n)
        @test sum(abs.(argsup_proba_risk(prob,AVaR(0),1:n)-argsup_proba_risk(prob,WorstCase(),1:n)) .<= EPSILON*ones(n)) == n
    end

    @testset "Equality Expectation AVaR(1)" begin
        n = 100
        prob = 1/n*ones(n)
        @test sum(abs.(argsup_proba_risk(prob,AVaR(1),1:n)-argsup_proba_risk(prob,Expectation(),1:n)) .<= EPSILON*ones(n)) == n
    end

    @testset "Equality Expectation ConvexCombi(beta,1)" begin
        n = 100
        prob = 1/n*ones(n)
        @test sum(abs.(argsup_proba_risk(prob,Expectation(),1:n)-argsup_proba_risk(prob,ConvexCombi(rand(),1),1:n)) .<= EPSILON*ones(n)) == n
    end

    @testset "Equality AVaR(beta) ConvexCombi(beta,0)" begin
        n = 100
        beta = rand()
        prob = 1/n*ones(n)
        @test sum(abs.(argsup_proba_risk(prob,AVaR(beta),1:n)-argsup_proba_risk(prob,ConvexCombi(beta,0),1:n)) .<= EPSILON*ones(n)) == n
    end

    # Check that in the case of minimization, AVaR find the worst costs
    @testset "Right sense of AVaR" begin
        betamax = rand()/2
        betamin = rand()/2+0.5
        n = 100
        prob = 1/n*ones(n)
        @test argsup_proba_risk(prob, AVaR(betamin), 1:n)'*(n:-1:1) - argsup_proba_risk(prob, AVaR(betamax), 1:n)'*(n:-1:1) >= 0
    end
end

# AVaR can be computed as a linear program
# We check equality between the formulation of Rockafellar and Urysev
@testset "Equality AVaR linear program" begin
    n = 100
    X = rand(-100:100,100)
    beta = rand()
    prob = 1/n*ones(n)

    m = Model(solver = ClpSolver())

    @variable(m, alpha)
    @variable(m,theta[1:n] >= 0)

    @constraint(m, theta[1:n] .>= X[1:n] - alpha)

    @objective(m, Min, alpha + 1/(beta)*sum(prob[i]*theta[i] for i in 1:n))

    status = solve(m)

    probaAVaR = argsup_proba_risk(prob, AVaR(beta), X)

    @test abs(probaAVaR'*X - getobjectivevalue(m)) <= EPSILON
end
