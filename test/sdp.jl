################################################################################
# Test SDDP functions
################################################################################
using FactCheck, StochDynamicProgramming
include("../src/SDPutils.jl")
using SDPutils


facts("Indexation for SDP") do

    bounds = [(0.1,10.0), (1.2, 4.0), (0.5, 2.0)]
    steps = [0.1, 0.05, 0.01]
    var = [0.4, 3.7, 1.9]
    vart = [0.42, 3.78, 1.932]

    ind = SDPutils.index_from_variable(var, bounds, steps)
    ind2 = SDPutils.real_index_from_variable(vart, bounds, steps)


    @fact ind --> (4,51,141)
    @fact ind2[1] --> roughly(4.2)
    @fact ind2[2] --> roughly(52.6)
    @fact ind2[3] --> roughly(144.2)


end


facts("SDP algorithm") do

    # Number of timesteps :
    TF = 3

    # Capacity of dams:
    VOLUME_MAX = 20
    VOLUME_MIN = 0

    # Specify the maximum flow of turbines:
    CONTROL_MAX = 10
    CONTROL_MIN = -10

    # Some statistics about aleas (water inflow):
    W_MAX = 5
    W_MIN = 0
    DW = 1

    T0 = 1

    # Define aleas' space:
    N_ALEAS = Int(round(Int, (W_MAX - W_MIN) / DW + 1))
    ALEAS = linspace(W_MIN, W_MAX, N_ALEAS);

    N_CONTROLS = 2;
    N_STATES = 2;
    N_NOISES = 1;

    infoStruct = "HD"

    COST = 66*2.7*(1 + .5*(rand(TF) - .5));

    # Define dynamic of the dam:
    function dynamic(t, x, u, w)
        return [x[1] + u[1] + w[1] - u[2], x[2] - u[1]]
    end

    # Define cost corresponding to each timestep:
    function cost_t(t, x, u, w)
        return COST[t] * (u[1])
    end

    function constraints(t, x, u, w)
        return (x[1]<=VOLUME_MAX)&(x[1]>=VOLUME_MIN)&(x[2]<=VOLUME_MAX)&(x[2]>=VOLUME_MIN)
    end

    function finalCostFunction(x)
        return 0.
    end

    """Build admissible scenarios for water inflow over the time horizon."""
    function build_scenarios(n_scenarios::Int64, N_STAGES)
        scenarios = zeros(n_scenarios, N_STAGES)

        for scen in 1:n_scenarios
            scenarios[scen, :] = (W_MAX-W_MIN)*rand(N_STAGES)+W_MIN
        end
        return scenarios
    end

    """Build probability distribution at each timestep based on N scenarios.
    Return a Vector{NoiseLaw}"""
    function generate_probability_laws(N_STAGES, N_SCENARIOS)
        aleas = zeros(N_SCENARIOS, N_STAGES, 1)
        aleas[:, :, 1] = build_scenarios(N_SCENARIOS, N_STAGES)

        laws = Vector{NoiseLaw}(N_STAGES)

        # uniform probabilities:
        proba = 1/N_SCENARIOS*ones(N_SCENARIOS)

        for t=1:N_STAGES
            aleas_t = reshape(aleas[:, t, :], N_SCENARIOS, 1)'
            laws[t] = NoiseLaw(aleas_t, proba)
        end

        return laws
    end

    N_SCENARIO = 10
    aleas = generate_probability_laws(TF-1, N_SCENARIO)

    x_bounds = [(VOLUME_MIN, VOLUME_MAX), (VOLUME_MIN, VOLUME_MAX)];
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (VOLUME_MIN, VOLUME_MAX)];

    x0 = [5, 0]

    alea_year = Array([7.0 7.0])

    aleas_scen = zeros(2, 1, 1)
    aleas_scen[:, 1, 1] = alea_year;

    stateSteps = [1,1];
    controlSteps = [1,1];
    monteCarloSize = 2;

    modelSDP = StochDynProgModel(TF, x_bounds, u_bounds,
                                    x0, cost_t,
                                    finalCostFunction, dynamic,
                                    constraints, aleas);

    paramsSDP = StochDynamicProgramming.SDPparameters(modelSDP, stateSteps,
                                                        controlSteps,
                                                        infoStruct,
                                                        "Exact");


        context("Compare StochDynProgModel constructors") do


            modelSDPPiecewise = StochDynamicProgramming.PiecewiseLinearCostSPmodel(TF,
            u_bounds, x0,
            [cost_t],
            dynamic, aleas)
            set_state_bounds(modelSDPPiecewise, x_bounds)

            modelSDPLinear = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(TF,
            u_bounds, x0,
            cost_t,
            dynamic, aleas)

            set_state_bounds(modelSDPLinear, x_bounds)


            test_costs = true
            x = x0
            u = [1, 1]
            w = [4]

            convertedSDPmodel = StochDynamicProgramming.build_sdpmodel_from_spmodel(modelSDPPiecewise)

            set_state_bounds(modelSDPLinear, x_bounds)


            for t in 1:TF-1
                test_costs &= (modelSDPLinear.costFunctions(t,x,u,w)==modelSDP.costFunctions(t,x,u,w))
                test_costs &= (modelSDPPiecewise.costFunctions[1](t,x,u,w)==modelSDP.costFunctions(t,x,u,w))
                test_costs &= (modelSDPPiecewise.costFunctions[1](t,x,u,w)==convertedSDPmodel.costFunctions(t,x,u,w))
            end

            @fact test_costs --> true

            @fact convertedSDPmodel.constraints(1,x,u,w) --> true

        end

        context("Solve and simulate using SDP") do
            paramsSDP.infoStructure = "anything"
            @fact_throws solve_DP(modelSDP, paramsSDP, false);
            paramsSDP.infoStructure = infoStruct

            V_sdp = solve_DP(modelSDP, paramsSDP, false);

            @fact size(V_sdp) --> (paramsSDP.stateVariablesSizes..., TF)

            costs_sdp, stocks_sdp, controls_sdp = StochDynamicProgramming.sdp_forward_single_simulation(modelSDP,
                                                                                                        paramsSDP,
                                                                                                        aleas_scen, x0,
                                                                                                        V_sdp, true )


            costs_sdp2, stocks_sdp2, controls_sdp2 = StochDynamicProgramming.sdp_forward_simulation(modelSDP,
                                                                                                    paramsSDP,
                                                                                                    aleas_scen,
                                                                                                    V_sdp, true )

            @fact costs_sdp2[1] --> costs_sdp

            x = x0
            V_sdp2 = StochDynamicProgramming.sdp_compute_value_functions(modelSDP, paramsSDP, false);
            paramsSDP.infoStructure = "DH"
            V_sdp3 = StochDynamicProgramming.sdp_compute_value_functions(modelSDP, paramsSDP, false);
            paramsSDP.infoStructure = "HD"

            Vitp = StochDynamicProgramming.value_function_interpolation( modelSDP.dimStates, V_sdp, 1)
            Vitp2 = StochDynamicProgramming.value_function_interpolation( modelSDP.dimStates, V_sdp2, 1)
            Vitp3 = StochDynamicProgramming.value_function_interpolation( modelSDP.dimStates, V_sdp3, 1)

            v1 = Vitp[(1.1,1.1)...]
            v2 = Vitp2[(1.1,1.1)...]
            v3 = Vitp3[(1.1,1.1)...]

            @fact v1 --> v2
            @fact (v1<=v3) --> true

            paramsSDP.infoStructure = "DH"
            costs_sdp3, stocks_sdp3, controls_sdp3 = StochDynamicProgramming.sdp_forward_simulation(modelSDP,
                                                                                                    paramsSDP,
                                                                                                    aleas_scen,
                                                                                                    V_sdp3, true )
            paramsSDP.infoStructure = "HD"

            @fact costs_sdp3[1]>=costs_sdp2[1] --> true

            a,b = StochDynamicProgramming.generate_grid(modelSDP, paramsSDP)

            x_bounds = modelSDP.xlim
            x_steps = paramsSDP.stateSteps

            u_bounds = modelSDP.ulim
            u_steps = paramsSDP.controlSteps

            @fact length(collect(a)) --> (x_bounds[1][2]-x_bounds[1][1]+x_steps[1])*(x_bounds[2][2]-x_bounds[2][1]+x_steps[2])/(x_steps[1]*x_steps[2])
            @fact length(collect(b)) --> (u_bounds[1][2]-u_bounds[1][1]+u_steps[1])*(u_bounds[2][2]-u_bounds[2][1]+u_steps[2])/(u_steps[1]*u_steps[2])

            ind = SDPutils.index_from_variable(x, x_bounds, x_steps)
            @fact get_bellman_value(modelSDP, paramsSDP, V_sdp2) --> V_sdp2[ind...,1]

            @fact size(V_sdp) --> (paramsSDP.stateVariablesSizes..., TF)
            @fact V_sdp2[1,1,1] <= V_sdp3[1,1,1] --> true

            @fact size(stocks_sdp) --> (3,1,2)
            @fact size(controls_sdp) --> (2,1,2)

            state_ref = zeros(2)
            state_ref[1] = stocks_sdp[2,1,1]
            state_ref[2] = stocks_sdp[2,1,2]
            w = [4]

            @fact_throws get_control(modelSDP,paramsSDP,V_sdp3, 1, x)
            @fact (get_control(modelSDP,paramsSDP,V_sdp3, 1, x, w)[1] >= CONTROL_MIN) --> true
            @fact (get_control(modelSDP,paramsSDP,V_sdp3, 1, x, w)[1] <= CONTROL_MAX) --> true

            paramsSDP.infoStructure = "DH"
            @fact (get_control(modelSDP,paramsSDP,V_sdp3, 1, x)[1] >= CONTROL_MIN) --> true
            @fact (get_control(modelSDP,paramsSDP,V_sdp3, 1, x)[1] <= CONTROL_MAX) --> true

            @fact size(stocks_sdp) --> (3,1,2)
            @fact size(controls_sdp) --> (2,1,2)

        end

    end
