# Implementation of Kelley's algorithm in Julia

using MathProgBase
using PyPlot
using Clp

"""
Compute the value and the derivative of the square function at
position `z`.

"""
function oracle(x::Float64)
    return x^2, 2x;
end


"""
Perform the kelley algorithm with the function corresponding to `oracle`.

"""
function kelley(xmin::Float64=-10.1, xmax::Float64=12.3, niter::Int64=10)
    # V will store coordinates of each cuts:
    V = zeros((niter, 3));

    V[:, 2] = - Inf;
    c = [1, 0]

    # Instantiate the algorithm at the first value encountered:
    y, z = oracle(xmin);
    V[1, 1] = z;
    V[1, 2] = y - z*xmin;
    V[1, 3] = xmin;

    # constraints matrix:
    A = zeros(niter + 2, 2)
    A[1:3, 1:2] = [0 -1; 0 1; -1 V[1, 1]]

    # constraints vector:
    b = zeros(niter+2)
    b[1:3] = [-xmin,xmax,-V[1, 2]]

    # Find niter cuts at each iteration:
    for k=2:niter
        # Solve linear program with MathProgBase:
        sol = linprog(c, A,'<',b, -Inf, Inf, ClpSolver())
        # Get optimal solution found:
        u_opt = sol.sol

        # Update the constraints with this new cuts:
        y, z = oracle(u_opt[2])
        V[k, 1] = z
        V[k, 2] = y - z * u_opt[2]
        V[k, 3] = u_opt[2]

        # update constraints:
        A[k+2, :] = [-1 V[k, 1]]
        b[k+2] = -V[k, 2]
    end
    return V
end

"""
Test if Kelley's algorithm is working.
"""
function main()

    # get positions of all cuts:
    V = kelley()

    # Define a x axis to plot result:
    xaxis = -10:.1:10

    # Define square function and vectorize it:
    square(x) = x^2
    @vectorize_1arg Number square

    # Plot with matplotlib:
    plot(xaxis, square(xaxis), lw=2, c="k")
    for nn in 1:10
        plot(xaxis, V[nn, 1]*xaxis + V[nn, 2], c="red")
    end

    ylim(-25, 150)
    show()
end
