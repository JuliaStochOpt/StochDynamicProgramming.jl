#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Some useful methods
#
#############################################################################

function costFunction(t,x,u,xi)
    #TODO
    	Cx 	= [3 4 2; 3 4 2];
	Cu 	= [2 1 1; 2 1 1];
	Cxi 	= [1 4 5; 1 4 5];

	lengthx  = length(Cx[:,t]);
	lengthu  = length(Cu[:,t]);
	lengthxi = length(Cxi[:,t]);

	cost 	= [Cx[:,t];Cu[:,t];Cxi[:,t];1.0];

	return cost
end

function dynamic(t,x,u,xi)
    #TODO
	A1        = [1.0 0 -1.0 0 -1.0 0 0.0 ; 0 1.0 0 -1.0 -1.0 0 0.0];
	A2        = [1.0 0 -1.0 0 -1.0 0 0.0 ; 0 1.0 0 -1.0 0 -1.0 0.0];
	A3 	  = []; #TODO integrate the fact that there is no dynamic for the final step
	dynamique = Any[A1',A2'];

	new_state = dynamique[t]';
    	
	return new_state
end
