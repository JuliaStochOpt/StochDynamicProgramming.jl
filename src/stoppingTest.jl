"""
Test if the stopping criteria is fulfilled.

Return true if |V0 - upb|/V0 < epsilon

# Arguments
* `V0::Float`:
    Approximation of initial cost
* `upb::Float`:
    Approximation of the upper bound given by Monte-Carlo estimation
*  `epsilon::Float`:
    Sensibility

# Return
`Bool`
"""
function test_stopping_criterion(V0::Float64, upb::Float64, epsilon::Float64)
    return abs((V0-upb)/V0) < epsilon
end
