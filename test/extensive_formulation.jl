################################################################################
# Test extensive formulation
################################################################################

using StochDynamicProgramming, Base.Test, Clp

@testset "Extensive formulation" begin
    solver = ClpSolver()

    # SDDP's tolerance:
    epsilon = .05
    # maximum number of iterations:
    max_iterations = 2
    # number of scenarios in forward and backward pass:
    n_scenarios = 10
    # number of aleas:
    n_aleas = 5
    # number of stages:
    n_stages = 3

    # define dynamic:
    function dynamic(t, x, u, w)
        return [x[1] - u[1] - u[2] + w[1]]
    end
    # define cost:
    function cost(t, x, u, w)
        return -u[1]
    end

    # Generate probability laws:
    laws = Vector{NoiseLaw}(n_stages)
    proba = 1/n_aleas*ones(n_aleas)
    for t=1:n_stages
        laws[t] = NoiseLaw([0, 1, 3, 4, 6], proba)
    end

    # set initial position:
    x0 = [10.]
    u_bounds = [(0., 7.), (0., Inf)]

    model_ef = StochDynamicProgramming.LinearSPModel(n_stages, u_bounds,
                                                        x0, cost, dynamic, laws)
    x_bounds_ef = [(0, 100)]
    set_state_bounds(model_ef, x_bounds_ef)

    # Instantiate parameters of SDDP:
    params = StochDynamicProgramming.SDDPparameters(solver,
                                                    passnumber=n_scenarios,
                                                    gap=epsilon,
                                                    max_iterations=max_iterations)

    @testset "Extensive solving" begin
        ef_cost = StochDynamicProgramming.extensive_formulation(model_ef, params)[1]
        @test isa(ef_cost, Float64)
    end

    @testset "Unsolvable extensive formulation" begin
        x_bounds_ef = [(-2., -1.)]
        set_state_bounds(model_ef, x_bounds_ef)
        @test_throws ErrorException extensive_formulation(model_ef, params)
    end
end
