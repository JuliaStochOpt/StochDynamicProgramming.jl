#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  the actual optimization function 
#
#############################################################################

function optimize()
    
    # TODO initialization
    
    stopping_test::Bool = false;
    iteration_count::int = 0;
    
    while(!stopping_test)
        stockTrajectories = forwardSimulations(forwardPassNumber, 
                            returnCosts = false,  
                            returnStocks=true, 
                            returnControls= false);
        backwardPass(stockTrajectories);
        #TODO stopping test
        
        iteration_count+=1;
    end
end
