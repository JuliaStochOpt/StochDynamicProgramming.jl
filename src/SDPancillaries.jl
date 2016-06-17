
module SDPancil
    using Interpolations

    export index_from_variable_1, real_index_from_variable_1,
            compute_V_given_t_DH

    function index_from_variable_1( variable,
                        bounds::Array,
                        variable_steps::Array)

        return tuple([ 1 + floor(Int64,(1e-10+( variable[i] - bounds[i][1] )/ variable_steps[i] )) for i in 1:length(variable)]...)
    end

    """
    Convert the state and control float tuples (stored as arrays or tuples) of the
    problem into float tuples that can be used as indexes for the interpolated
    value function

    Parameters:
    - variable (Array)
        the vector variable we want to convert to an index (integer)

    - bounds (Array)
        the lower bounds for each component of the variable

    - variable_steps (Array)
        discretization step for each component


    Returns :
    - index (tuple of integers)
        the indexes of the variable
    """
    function real_index_from_variable_1( variable,
                        bounds::Array,
                        variable_steps::Array)

        return tuple([1 + ( variable[i] - bounds[i][1] )/variable_steps[i] for i in 1:length(variable)]...)
    end

    function compute_V_given_x_t_DH(sampling_size, samples, probas, u_bounds, x_bounds, x_steps, x_dim,
                                    product_controls, dynamics, constraints, cost, V, Vitp, t, x)
        expected_V = Inf
        optimal_u = tuple()
        current_cost = 0
        #Loop over controls
        for u = product_controls

            expected_V_u = 0.
            count_admissible_w = 0

            for w = 1:sampling_size
                w_sample = samples[:, w]
                proba = probas[w]
                next_state = dynamics(t, x, u, w_sample)

                if constraints(t, next_state, u, w_sample)

                    count_admissible_w = count_admissible_w + proba
                    ind_next_state = real_index_from_variable_1(next_state, x_bounds, x_steps)
                    next_V = Vitp[ind_next_state...]
                    current_cost = cost(t, x, u, w_sample)
                    expected_V_u += proba*(current_cost + next_V)

                end
            end

            if (count_admissible_w>0)

                next_V = next_V / count_admissible_w

                if (expected_V_u < expected_V)

                    expected_V = expected_V_u
                    optimal_u = u

                end
             end
        end
        ind_x = index_from_variable_1(x, x_bounds, x_steps)

        V[ind_x..., t] = expected_V
        #return(expected_V)
    end

    function compute_V_given_x_t_u_DH(sampling_size, samples, probas, x_bounds, x_steps,
                                        dynamics, constraints, cost, Vitp, t, x, expected_Vs, u)

        expected_V_u = 0.
        count_admissible_w = 0
        current_worker = myid() - (nprocs()>1)

        for w = 1:sampling_size
            w_sample = samples[:, w]
            proba = probas[w]
            next_state = dynamics(t, x, u, w_sample)

            if constraints(t, next_state, u, w_sample)

                count_admissible_w = count_admissible_w + proba
                ind_next_state = real_index_from_variable_1(next_state, x_bounds, x_steps)
                expected_V_u += proba*(cost(t, x, u, w_sample) + Vitp[ind_next_state...])

            end
        end

        if (count_admissible_w>0)

            if (expected_V_u < expected_Vs[current_worker])

                expected_Vs[current_worker] = expected_V_u

            end
        end

    end

end