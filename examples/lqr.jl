# Implement a linear quadratic feedback controller.

# Source: Bertsekas, Dynamic Programming and Optimal Control

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
function solve_riccati_backward(Tf, A, B, Q, R, Sf)
    # This array will store evolution of Riccati gain;=:
    L = zeros(Tf, 1, 2)
    P = Sf

    for i=(Tf-1):-1:1
        K = -inv(R + B'*P*B)*B'*P*A
        P = Q + A'*P*A + A'*P*B*K
        L[i, :, :] = K
    end

    return L
end



"""

Test Riccati equation upon a toy example.

"""
function main(rho = .3)

    Tf = 20
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
    # c(x,u) = x' Q x + u' R u
    R = rho * eye(1)
    Q = C'*C

    # Solve Riccati equation:
    L = solve_riccati_backward(Tf, A, B, Q, R, Q)
    # Instantiate x at time 0:
    x = [1;0]

    # Simulate system's evolution and apply LQR control:
    for i=1:Tf
        # Get gain stored in L:
        K = reshape(L[i, :, :], 1, 2)
        # control is straightforward:
        u = K*x
        # Simulate system's evolution:
        x = A*x + B*K*x
        # ... and get observation:
        y[i] = (C*x)[1]
        ctrl[i] = u[1]
    end

    # Plot result with matplotlib:
    step(t, y)
    step(t, ctrl)
    show()
end