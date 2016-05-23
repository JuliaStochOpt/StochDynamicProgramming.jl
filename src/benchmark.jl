#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Compare the optimal values and control returned by different instances
#  of SDDP on the same problem
#############################################################################

"""
Create different sets of parameters from a problem and compare the results
of these different instances.

"""
benchmark(model::SPmodel,
          SDDParametersCollection,
          seeds)



#return a table with
#(computing time, Gap, SDDPiterationNumber,CPLEXcalls)

end
