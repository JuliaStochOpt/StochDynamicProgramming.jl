#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Model and solve the One-Step One Alea problem in different settings
# - used to compute the optimal control (in forward phase / simulation)
# - used to compute the cuts in the Backward phase
#############################################################################

function solveOneStepOneAlea(t::int,
                                    x,
                                    xi,
                                    returnOptNextStep::Bool=false, 
                                    returnOptControl::Bool=false,
                                    returnSubgradient::Bool=false,
                                    returnCost::Bool=false)
    
    #TODO call the right following function
    # return (optNextStep, optControl, subgradient, cost) #depending on which is asked
end

function solveOneStepOneAleaLinear(t::int,
                                    x, #TODO type
                                    xi,#TODO type
                                    returnOptNextStep::Bool=false, 
                                    returnOptControl::Bool=false,
                                    returnSubgradient::Bool=false,
                                    returnCost::Bool=false)
    #TODO call the solver on the linear problem
    #TODO return optNextStep, optControl, optValue, subgradient 
end
