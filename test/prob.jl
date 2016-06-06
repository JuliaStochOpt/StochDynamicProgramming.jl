################################################################################
# Test probability functions
################################################################################
using FactCheck, StochDynamicProgramming

facts("Probability functions") do
    support = [1, 2, 3]
    proba = [.2 .5 .3]

    law = NoiseLaw(support, proba)
    @fact typeof(law) --> NoiseLaw
    @fact law.supportSize --> 3

    dims = (2, 2, 1)
    scenarios = simulate_scenarios([law, law], 2)
    @fact typeof(scenarios) --> Array{Float64, 3}
    @fact size(scenarios) --> (2, 2, 1)

    # test product of noiselaws:
    support2 = [4, 5, 6]
    proba2 = [.3 .3 .4]
    law2 = NoiseLaw(support2, proba2)
    law3 = StochDynamicProgramming.noiselaw_product(law, law2)
    @fact law3.supportSize --> law.supportSize*law2.supportSize
    @fact law3.proba --> vec(proba' * proba2)
    @fact size(law3.support)[1] --> size(law.support)[1] + size(law2.support)[1]
    @fact law3.support[:, 1] --> [1., 4.]

    # Test product of three noiselaws:
    StochDynamicProgramming.noiselaw_product(law, law2, law)

    # Test sampling:
    samp = StochDynamicProgramming.sampling([law, law2, law3], 1)
end

