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
end

