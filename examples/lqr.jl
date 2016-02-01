#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Implement a linear quadratic feedback controller.
# Source: Bertsekas, Dynamic Programming and Optimal Control
#############################################################################





using PyPlot


"""
Solve Riccati equation with backward recursion.

The problem to solve is the following (quadratic linear settings):

```
x(t+1) = A * x(t) + B * u(t)

```

with cost:

```
J(t) = x' Q x + u' R u

```

Parameters:
    - `Tf`: final time.
    - `Sf`: final cost.

Returns:
    - `L`: evolution of gains, indexed by time.

"""
function solve_riccati_backward(Tf::Int64,
                                A, #::Array{Float64, 2},
                                B, #::Array{Float64, 2},
                                Q, #::Array{Float64, 2},
                                R, #::Array{Float64, 2},
                                Sf,) #::Array{Float64, 2})
    # This array will store evolution of Riccati gain;=:
    L = zeros(Tf, 1, 2)
    P_array = zeros(Tf, 2, 2)
    P = Sf

    for i=(Tf-1):-1:1
        K = -inv(R + B'*P*B)*B'*P*A
        P = Q + A'*P*A + A'*P*B*K
        L[i, :, :] = K
        P_array[i, :, :] = P
    end

    return L, P_array
end



"""

Test Riccati equation upon a toy example.

"""
function simulate_lqr(Tf=20, rho = .3, sigma=.01)

    t = 1:Tf
    y = zeros(Tf)
    ctrl = zeros(Tf)

    # Define system's matrix:
    # x(t+1) = A x(t) + B u(t)
    A = [1 1; 0 1]
    B = [0; 1]

    # y = C x
    C = [1 0]

    # Define cost matrix corresponding to the following cost:
    # c(x,u) = x' Q x + u' R uc'e
    R = rho * eye(1)
    Q = C'*C

    # Solve Riccati equation:
    L, P_arr = solve_riccati_backward(Tf, A, B, Q, R, Q)
    # Instantiate x at time 0:
    x = [0;0]

    total_cost = 0
    theoric_cost = x'* reshape(P_arr[1, :, :], 2, 2) * x

    # Simulate system's evolution and apply LQR control:
    for i=1:Tf-1
        # Get gain stored in L:
        K = reshape(L[i, :, :], 1, 2)
        # control is straightforward:
        u = K*x
        # Simulate perturbation:
        w = sigma * randn(2)
        # Simulate system's evolution:
        total_cost += (x'*Q*x + u'*R*u)[1]
        x = A*x + B*K*x + w
        # ... and get observation:
        theoric_cost += sigma^2 * [1,1]'*reshape(P_arr[i+1, :, :], 2, 2)*[1,1]
        y[i] = (C*x)[1]
        ctrl[i] = u[1]
    end

    # Plot result with matplotlib:
    # step(t, y)
    # step(t, ctrl)
    # show()
    return total_cost, theoric_cost
end

function benchmark_lqr(n_simu=100::Int64)

    costs = zeros(n_simu)

    for n=1:n_simu
        costs[n] = simulate_lqr(20, .3, 0.01)[1]
    end

    println(mean(costs))
    hist(costs)
    show()
end
