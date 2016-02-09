
push!(LOAD_PATH, "../examples")

using Base.Test
using lqr

function test_lqr()
    rho = .3
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

    @test_approx_eq_eps norm(x) 0 1e-6
    @test_approx_eq_eps ctrl[end] 0 1e-6
end
