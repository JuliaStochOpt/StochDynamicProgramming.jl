.. _sddp_api:

========================
SDDP: SPModel and solver
========================

We detail here the interface with:

- the object `SPModel` that stores the implementation of the stochastic problem,
- the SDDP solver `SDDPparameters`.


LinearSPModel
=============

`LinearSPModel` implements the stochastic problem we want to tackle. Currently
the package supports only linear dynamic (as we remain in the settings of SDDP)
but allows to use either piecewise linear or quadratic costs.

To define a `LinearSPModel`, the constructor is::

    spmodel = LinearSPModel(
                        nstage,             # number of stages
                        ubounds,            # bounds of control
                        x0,                 # initial state
                        cost,               # cost function
                        dynamic,            # dynamic
                        aleas;              # modelling of noises
                        Vfinal=nothing,     # final cost
                        eqconstr=nothing,   # equality constraints
                        ineqconstr=nothing, # inequality constraints
                        control_cat=nothing # category of controls
                           )

Default parameters
^^^^^^^^^^^^^^^^^^
You should at least specify these parameters to define a `LinearSPModel`:

- `nstage` (Int): number of stage in our stochastic multistage problem
- `ubounds` (list of tuple): the bounds upon control, defined as a sequence of tuple :code:`(umin, umax)`.
- `x0` (`Vec{Float64}`): initial position
- `cost` (`Function`): cost function
- `dynamic` (`Function`): system's dynamic
- `aleas` (`Vector{NoiseLaw}`): modelling of perturbations


State bound
^^^^^^^^^^^

By default, the state is not bounded. If it is needed to add some constraints upon
it, just call::

    set_state_bounds(spmodel, s_bounds)


Final cost function
^^^^^^^^^^^^^^^^^^^
By default, the final cost function is taken equal to 0. If this is not the
cast, you could define it with a dedicated function or a `PolyhedralFunction` object:

- `Vfinal` (`PolyhedralFunction` or `Function`): final cost (default is 0.)


Local equality and inequality constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, there are no equality or inequality constraints. If you want to add
some, you should add in arguments these constraints with a Function:

- `eqconstr` (`Function`): if necessary, add equality constraints to each timestep.
- `ineqconstr` (`Function`): if necessary, add inequality constraints to each timestep.


Mixed-Integer SDDP
^^^^^^^^^^^^^^^^^^

By default, SDDP handles only continuous controls. But a great deal of problem
must adress integer or binary controls. If so, you could specify to SDDP
to handle integer controls with this argument:

- `control_cat` (`Vec{Union{:Cont, :Bin}`): specify if necessary if some controls are binary

If integer controls are found, then forward passes will be computed ensuring that
specified controls are integer whereas these integrity
constraints is relaxed during backward passes.



SDDPparameters
==============

To define a `SDDPparameters`, the constructor is (with its default parameters)::

    function SDDPparameters(solver;
                            passnumber=10,
                            gap=0.,
                            max_iterations=20,
                            prune_cuts=0,
                            pruning_algo="none",
                            compute_ub=-1,
                            montecarlo_final=10000,
                            montecarlo_in_iter=100,
                            mipsolver=nothing,
                            rho0=0.,
                            alpha=1.)




Usual arguments
^^^^^^^^^^^^^^^

You could set the different parameters for SDDP with:

- `solver`: the solver used to solve inner LP problems in SDDP
- `passnumber`: the number of forward passes to perform at each iteration
- `gap`: the admissible gap between lower-bound and upper-bound to achieve convergence
- `max_iterations`: the maximum number of iterations to perform

By default, no upper-bound is computed as this computation is costly.
You can specify three different means to compute upper-bound:

- `compute_ub` is set to -1 if you do not want to compute upper-bound.
- If you want to compute upper-bound at the end of iteration, set it to 0.
- Otherwise, if you want to compute upper-bound every `n` iteration, set it to `n`.

  You can also specify the number of simulation
to use to compute upper-bound.
- `compute_ub`: specify when to compute upper-bounds
- `montecarlo_final`
- `montecarlo_in_iter`


Cuts pruning
^^^^^^^^^^^^

The more iterations are runned, the more cuts are stored. Sometime, it could be useful to remove
useless cuts to remove constraints in LP problems. Three different kind of cuts
prunings are implemented currently:

- exact cuts pruning
- level1 cuts pruning
- mixed level1/exact cuts pruning

To define how to use cuts pruning in `SDDPparameters`:
- `prune_cuts`
- `pruning_algo`

Remark: this feature is still in beta, and level1 cuts pruning could be
memory intensive due to the fact that all visited states are stored in
an array.


Quadratic regularization
^^^^^^^^^^^^^^^^^^^^^^^^

- `rho0`
- `alpha`


Mixed-Integer SDDP
^^^^^^^^^^^^^^^^^^

If the problem is a SMIP, you need to use a MILP solver (Cbc, Gurobi or CPLEX).
You should specify to the SDDP instance which solver to use with:
- `mipsolver`
