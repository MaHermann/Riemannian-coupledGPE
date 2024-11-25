using Random

include("../src/Riemannian-coupledGPE.jl")
using .MulticomponentBEC
## Model parameters
params = Dict([])
@parameters params begin
    dimension               = 2
    n_components            = 3
    masses                  = [1.0, 1.0, 1.0]
    interactions            = [ 0.5 1.0 1.0;
                                1.0 5.0 1.0;
                                1.0 1.0 10.0]
    domain                  = [0.0, 1.0]
    potential_type          = :periodic    # change to :random for the random potential case
    l_ε                     = 6
end
## Numerical parameters
@parameters params begin
    logging_active          = true
    n_elements              = 2^10    # per direction
    interp_degree           = 2
    quad_degree             = 5
    step_size               = 1.0
    ω                       = 1.0           # set to 0.99 for the regularized Newton_method
    optimization_algorithm  = Newton_method # alternating_gradient_descent_energy_adaptive or alternating_gradient_descent_Lagrangian
    preconditioner_type     = :A0normNewton # set this to :A0norm when using a gradient method instead of a Newton method
    solver_reltol           = 10
    solver_max_iter         = nothing
    start_value             = :constant
    start_residual          = 1e-4
    termination_residual    = 1e-8
    retraction              = normalization_retraction!
    random_seed             = 1552
    max_iter                = 3000
    initialzation_max_iter  = 100     # should never be relevant
end
##
Random.seed!(random_seed);
## Setup logging
n = size(generate_grid_context(
        dimension, n_elements, interp_degree, quad_degree;
        left = domain[1], right = domain[2],
    ).M, 1) # a bit wasteful but seems to be the cleanest way to get this info here
logging_callback = LoggingCallback(n, n_components, max_iter)
initialization_time = [0.0]
preconditioner_time = [0.0]
n_initialization_steps = [-1]
initialization_residuals = zeros(n_components)
log = Dict(
    "energies" => logging_callback.energies,
    "residual_values" => logging_callback.residual_values,
    "n_inner_iterations" => logging_callback.n_inner_iterations,
    "solver_resnorms" => logging_callback.solver_resnorms,
    "n_matrix_multiplications" => logging_callback.n_matrix_multiplications,
    "τs" => logging_callback.τs,
    "σs" => logging_callback.σs,
    "preconditioner_time" => preconditioner_time,
    "initialization_time" => initialization_time,
    "n_initialization_steps" => n_initialization_steps,
    "initialization_residuals" => initialization_residuals,
)
##
@experiment params log begin
    # Setup
    grid_context = generate_grid_context(dimension, n_elements, interp_degree,
        quad_degree; left = domain[1], right = domain[2],
    )
    L = assemble_stiffness_matrix(grid_context)
    confinement_potential = ConstantPotential(
        x -> 1e6 .* max((2*x[1] - 1)^40, (2*x[2] - 1)^40), grid_context,
    )
    if potential_type == :random
        random_potential = create_random_checkerboard_potential_2D(grid_context, l_ε)
        potential = add_potentials(random_potential, confinement_potential)
    else
        periodic_potential = create_periodic_potential_2D(grid_context, l_ε)
        potential = add_potentials(periodic_potential, confinement_potential)
    end
    gpe_system = GPESystem(L, potential, interactions, masses, grid_context)
    if start_value == :constant
        ϕ₀ = constant_normed_PFrame(gpe_system)
    else
        ϕ₀ = random_normed_PFrame(gpe_system)
    end
    residual_temp = zeros(n_components) # temp array for the termination_criterion
    # Preconditioning
    logging_active && (elapsed_time = Base.time_ns())
    preconditioner_initialization = get_A0preconditioner(gpe_system)
    if preconditioner_type == :A0norm
        preconditioner = preconditioner_initialization
    elseif preconditioner_type == :A0normNewton
        preconditioner = get_A0preconditioner(gpe_system; isNewton = true)
    elseif preconditioner_type == :nothing
         preconditioner = get_identitypreconditioner()
    end
    if logging_active
        elapsed_time = Base.time_ns() - elapsed_time
        preconditioner_time[1] = Float64(elapsed_time) / 1e9
    # Initialization
        count_steps(_) = (n_initialization_steps[1] += 1)
        elapsed_time = Base.time_ns()
    end
    ϕ₀ = alternating_gradient_descent_energy_adaptive(
        ϕ₀, gpe_system;
        max_iter = initialzation_max_iter,
        reltol_from_R = true,
        termination_criterion = SumResidual(
            start_residual, false, false, residual_temp,
        ),
        solver = CGSolver(
            eltype(ϕ₀[1]);
            preconditioner = preconditioner_initialization, isLogging = false,
        ),
        verbose = false, callback = logging_active ? count_steps : nothing,
    )
    if logging_active
        elapsed_time = Base.time_ns() - elapsed_time
        initialization_time[1] = Float64(elapsed_time) / 1e9
        residuals!(initialization_residuals, ϕ₀, gpe_system)
        gpe_system.n_matrix_multiplications[1] = 0
    end
    # Optimization
    ϕ = optimization_algorithm(
        ϕ₀, gpe_system;
        max_iter = max_iter, step_size = step_size, ω = ω,
        solver = CGSolver(
            eltype(ϕ₀[1]);
            preconditioner = preconditioner, reltol = solver_reltol, 
            maxiter = solver_max_iter, isLogging = logging_active,
        ),
        retraction = retraction,
        termination_criterion = SumResidual(
            termination_residual, false, true, residual_temp,
        ),
        verbose = false, callback = logging_active ? logging_callback : nothing,
    )
end
