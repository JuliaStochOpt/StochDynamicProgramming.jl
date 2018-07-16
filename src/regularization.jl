#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  SDDP regularization
#############################################################################


export SDDPRegularization

abstract type AbstractRegularization end


mutable struct SDDPRegularization <: AbstractRegularization
    ρ::Float64
    alpha::Float64
    incumbents
    decay::Float64
    function SDDPRegularization(rho0::Float64, alpha::Float64; decay=1.)
        return new(rho0, alpha, nothing, decay)
    end
end

function update_penalization!(reg::SDDPRegularization)
    reg.ρ *= reg.alpha
end

function getpenaltyexpr(reg::SDDPRegularization, x, xp)
    QuadExpr(reg.ρ*dot(x - xp, x - xp))
end

function push_state!(reg::SDDPRegularization, x::Vector, t::Int, k::Int)
    incumbents[t+1, k, :] = reg.decay*x + (1-reg.decay)*incumbents[t+1, k, :]
end

getincumbent(reg::SDDPRegularization, t::Int, k::Int) = reg.incumbents[t+1, k, :]
getavgincumbent(reg::SDDPRegularization, t::Int) = mean(reg.incumbents[t+1, :, :], 2)

