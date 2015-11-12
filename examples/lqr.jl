# Implement a linear quadratic feedback controller.
# Source: Bertsekas, Dynamic Programming and Optimal Control

using PyPlot

function solve_riccati_backward(n, A, B, Q, R, Sf)
    L = zeros(n, 1, 2)
    P = Sf

    for i=(n-1):-1:1
        K = -inv(R + B'*P*B)*B'*P*A
        P = Q + A'*P*A + A'*P*B*K
        L[i, :, :] = K
    end

    return L
end

function main(rho = .3)

    n = 20
    t = 1:n
    y = zeros(n)
    ctrl = zeros(n)

    # Define system's matrix:
    # x(t+1) = A x(t) + B u(t)
    A = [1 1; 0 1]
    B = [0; 1]

    # y = C x
    C = [1 0]

    # Define cost matrix:
    # c(x,u) = x' Q x + u' R u
    R = rho * eye(1)
    Q = C'*C

    # Solve Riccati equation:
    L = solve_riccati_backward(n, A, B, Q, R, Q)
    # Instantiate x at time 0:
    x = [1;0]

    # Simulate system's evolution and apply LQR control:
    for i=1:n
        K = reshape(L[i, :, :], 1, 2)
        u = K*x
        x = A*x + B*K*x
        y[i] = (C*x)[1]
        ctrl[i] = u[1]
    end

    # Plot result with matplotlib:
    step(t, y)
    step(t, ctrl)
    show()
end