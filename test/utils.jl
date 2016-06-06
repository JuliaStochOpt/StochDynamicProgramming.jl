################################################################################
# Test utils functions
################################################################################
using FactCheck, StochDynamicProgramming

facts("Utils functions") do
    # Test extraction of vector in array:
    arr = rand(4, 4, 2)
    v = StochDynamicProgramming.extract_vector_from_3Dmatrix(arr, 2, 1)
    @fact typeof(v) --> Vector{Float64}
    @fact size(v) --> (2,)
    @fact v --> vec(arr[2, 1,:])

    # Test upper bound calculation:
    cost = rand(10)
    upb = StochDynamicProgramming.upper_bound(cost)
    tol = sqrt(2) * erfinv(2*.975 - 1)
    @fact upb --> mean(cost) + tol*std(cost)/sqrt(length(cost))

    # Test stopping criterion:
    @fact StochDynamicProgramming.test_stopping_criterion(1., .999, 0.01) --> true
end

