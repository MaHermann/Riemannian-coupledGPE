using Random

include("../src/Riemannian-coupledGPE.jl")
using .MulticomponentBEC
## Model parameters
params = Dict([])
@parameters params begin
    dimension               = 1
    n_components            = 2
    α                       = 0.8
    β                       = 10
    masses                  = [α, (1-α)]
    interactions            = 2 * β *
                                [1.04 1.00;
                                 1.00 0.97]
    potential_function      = x -> 2 * (0.5 * x[1]^2 + 24 * cos(x[1]) * cos(x[1]))
    domain                  = [-16.0, 16.0]
end
## Numerical parameters
@parameters params begin
    logging_active          = true
    n_elements              = 2^10    # per direction
    interp_degree           = 2
    quad_degree             = 5
    step_size               = 1.0
    ω                       = 1.0     # set to 0.99 for the regularized Newton_method
    optimization_algorithm  = alternating_gradient_descent_energy_adaptive # change for other methods (see also localization.jl)
    preconditioner_type     = :A0norm # set this to :A0normNewton when using a Newton_method
    solver_reltol           = 1
    solver_max_iter         = nothing
    start_value             = :constant
    start_residual          = 1e-2
    termination_residual    = 1e-8
    retraction              = normalization_retraction!
    random_seed             = 1552
    max_iter                = 5000
    initalzation_max_iter   = 100     # should never be relevant
end
##
Random.seed!(random_seed);
##
# The time measurements here should not be taken too seriously, as for such fast 
# computations, most time is actually spent in the precompilation of the functions. 
# We could maybe avoid this with BenchmarkTools.jl etc.
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
    grid_context = generate_grid_context(
        dimension, n_elements, interp_degree, quad_degree;
        left = domain[1], right = domain[2],
    )
    L = assemble_stiffness_matrix(grid_context)
    potential = ConstantPotential(potential_function, grid_context);
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
        preconditioner = preconditioner_initialization;
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
        max_iter = initalzation_max_iter,
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
