#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Some useful methods
#
#############################################################################

using SDDP


"""
Instantiate a Polyhedral function corresponding to f -> 0

"""
function get_null_value_functions()
    V = SDDP.PolyhedralFunction(zeros(1), zeros(1, 1), 1)
    return V
end


"""
Extract a vector stored in a 3D Array


Parameters:
- input_array (Array{Float64, 3})
    array storing the values of vectors

- nx (Int64)
    Position of vector in first dimension

- ny (Int64)
    Position of vector in second dimension

Return:
- Vector{Float64}

"""
function extract_vector_from_3Dmatrix(input_array::Array{Float64, 3},
                                      nx::Int64,
                                      ny::Int64)

    state_dimension = size(input_array)[3]
    return reshape(input_array[ny, nx, :], state_dimension)
end


"""
Estimate the upper bound with the Monte-Carlo error


Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- V (bellmanFunctions)
    the current estimation of Bellman's functions

- forwardPassNumber (int)
    number of Monte-Carlo simulation

- returnMCerror (Bool)
    return or not the estimation of the MC error


Returns :
- estimated-upper bound

- Monte-Carlo error on the upper bound (if returnMCerror)

"""
function upper_bound(model::SDDP.SPModel,
                     param::SDDP.SDDPparameters,
                     V::Vector{SDDP.PolyhedralFunction},
                     forwardPassNumber::Int64,
                     returnMCerror::Bool)

    C = forward_simulations(model, param, V, forwardPassNumber,
                            nothing, true, false, false)
    m = mean(C)
    if returnMCerror
        return m, 1.96*std(C)/sqrt(forwardPassNumber)
    else
        return m
    end
end
