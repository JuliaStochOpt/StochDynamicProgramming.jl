#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Display SDDP simulations with matplotlib
#############################################################################

# WARNING: Matplotlib and PyCall must be installed!
using PyPlot

function display_stocks(model, stocks)

    ndims = size(stocks)[3]
    nsteps = model.stageNumber

    figure()
    for ind in 1:ndims
        subplot(ndims, 1, ind)
        plot(stocks[:, :, ind], lw=.5, color="k")
        xlim(0, nsteps)
        grid()
        ylabel(string("X", ind))
    end
    xlabel("Time")
end


function display_controls(model, controls)

    ndims = size(controls)[3]
    nsteps = model.stageNumber

    figure()
    for ind in 1:ndims
        subplot(ndims, 1, ind)
        plot(controls[:, :, ind], lw=.5, color="k")
        xlim(0, nsteps)
        grid()
        ylabel(string("U", ind))
    end
    xlabel("Time")
end


function display_costs_distribution(costs)
    figure()
    boxplot(costs, boxprops=Dict(:linewidth=>3, :color=>"k"))
    ylabel("Costs")
    grid()
end


function display_aleas(aleas)
    ndims = size(aleas)[3]
    nsteps = size(aleas)[1]

    figure()
    for ind in 1:ndims
        subplot(ndims, 1, ind)
        plot(1:nsteps, mean(aleas[:, :, ind], 2), lw=2, color="k")
        box = boxplot(aleas[:, :, ind]', widths=.25, boxprops=Dict(:color=>"k", :marker=>"+"))
        xlim(0, nsteps)
        grid()
        ylabel(string("\$W_", ind, "\$"))
    end
    xlabel("Time")

end


function display_execution_time(exectime)
    nit = size(exectime)[1]

    figure()
    plot(exectime, lw=.5, color="k")
    grid()
    xlim(0, nit)
    xlabel("Iteration")
    ylabel("Execution time (s)")
end


function display_bounds(model)
    nit = size(model.upperbounds)[1]

    figure()
    plot(model.upperbounds, lw=2, color="r", label="Upper bound")
    plot(model.lowerbounds, lw=2, color="g", label="Lower bound")
    xlabel("Iteration")
    ylabel("Estimation of bounds")
    grid()
    legend()
end


function display_all(model, costs, stocks, controls, aleas)

    #= display_bounds(model) =#
    #= display_execution_time(model.texec) =#
    display_aleas(aleas)
    display_controls(model, controls)
    display_stocks(model, stocks)
end

