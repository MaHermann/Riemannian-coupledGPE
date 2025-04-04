function riemannian_gradient_descent(
    ϕ₀::PFrame{p},
    gpe_system,
    gradient_function!;
    allocate_matrix = false,
    kwargs...,
) where {p}
    arguments = Dict{Symbol, Any}(kwargs)

    retraction! = get_retraction(arguments)
    iteration = get_iteration(arguments)
    termination_criterion = get_termination_criterion(arguments)
    callback = get_callback(arguments)
    step_size = get_step_size(arguments)
    solver = get_solver(arguments)
    gradient_kwargs = get_gradient_kwargs(arguments)
    if allocate_matrix
        variable_storage = VariableStorage(ϕ₀; gpe_system.grid_context)
    else
        variable_storage = VariableStorage(ϕ₀)
    end

    # preallocate and initialize
    ϕ = deepcopy(ϕ₀)
    riemannian_gradient = PFrame(SVector{p}(zero(ϕ₀[i]) for i in 1:p))
    n_step = [0]                # array to have fixed memory location
    τ = zeros(p)
    variable_storage.variables["ϕ"] = ϕ
    variable_storage.variables["riemannian_gradient"] = riemannian_gradient
    variable_storage.variables["gpe_system"] = gpe_system
    variable_storage.variables["n_step"] = n_step
    variable_storage.variables["τ"] = τ
    variable_storage.variables["solver"] = solver

    update!(gpe_system, ϕ)

    loop_concurrently!(
        ϕ, riemannian_gradient, gradient_function!, gpe_system, iteration, 
        callback, termination_criterion, variable_storage, step_size, retraction!,
        solver, n_step, τ, gradient_kwargs,
    )

    return ϕ
end

"""
    update_hamiltonians_at_each_step: if true, recompute the hamiltonians to one 
        common state at the beginning of each iteration. This causes a slight
        computational overhead, but is necessary e.g. for the correct computation
        of the residual as a termination criterion
"""
function alternating_gradient_descent(
    ϕ₀::PFrame{p},
    gpe_system,
    gradient_function!;
    allocate_matrix = false,
    update_hamiltonian_at_each_step = true,
    kwargs...,
) where {p}
    arguments = Dict{Symbol, Any}(kwargs)

    retraction! = get_retraction(arguments)
    iteration = get_iteration(arguments)
    termination_criterion = get_termination_criterion(arguments)
    callback = get_callback(arguments)
    step_size = get_step_size(arguments)
    solver = get_solver(arguments)
    gradient_kwargs = get_gradient_kwargs(arguments)
    if allocate_matrix
        variable_storage = VariableStorage(ϕ₀; gpe_system.grid_context)
    else
        variable_storage = VariableStorage(ϕ₀)
    end
    
    # preallocate and initialize
    ϕ = deepcopy(ϕ₀)
    riemannian_gradient = PFrame(SVector{p}(zero(ϕ₀[i]) for i in 1:p))
    n_step = [0]                # array to have fixed memory location
    τ = zeros(p)
    variable_storage.variables["ϕ"] = ϕ
    variable_storage.variables["riemannian_gradient"] = riemannian_gradient
    variable_storage.variables["gpe_system"] = gpe_system
    variable_storage.variables["n_step"] = n_step
    variable_storage.variables["τ"] = τ
    variable_storage.variables["solver"] = solver

    update!(gpe_system, ϕ)

    loop_alternatingly!(
        ϕ, riemannian_gradient, gradient_function!, gpe_system, iteration, callback, 
        termination_criterion, variable_storage, step_size, retraction!, solver,
        n_step, τ, gradient_kwargs;
        update_hamiltonian_at_each_step,
    )
    !update_hamiltonian_at_each_step && update_hamiltonian!(gpe_system)
    return ϕ
end

# Function barriers for type stabillity

function loop_concurrently!(
    ϕ::PFrame{p}, riemannian_gradient, gradient_function!, gpe_system, iteration, 
    callback, termination_criterion, variable_storage, step_size, retraction!,
    solver, n_step, τ, gradient_kwargs,
) where {p}
    for step in iteration
        n_step[1] = step

        if is_met(termination_criterion, ϕ, gpe_system; variable_storage)
            !isnothing(callback) && call!(callback, variable_storage)
            break
        end

        riemannian_gradient = gradient_function!(
            riemannian_gradient, ϕ, gpe_system;
            solver, variable_storage, update_gpe_system = false, gradient_kwargs...,
        )
        for i in 1:p
            τ[i] = determine(
                step_size, ϕ, gpe_system, riemannian_gradient, i; 
                retraction!, variable_storage,
            )
            lmul!(-1 * τ[i], riemannian_gradient[i])
        end
        retraction!(
            ϕ, riemannian_gradient, gpe_system;
            variable_storage, update_gpe_system = false,
        )
        update!(gpe_system, ϕ)
        !isnothing(callback) && call!(callback, variable_storage)
    end
end

function loop_alternatingly!(
    ϕ::PFrame{p}, riemannian_gradient::PFrame{p}, gradient_function!, gpe_system,
    iteration, callback, termination_criterion, variable_storage, step_size, 
    retraction!, solver, n_step, τ, gradient_kwargs;
    update_hamiltonian_at_each_step = true,
) where {p}
    for step in iteration
        n_step[1] = step

        if is_met(termination_criterion, ϕ, gpe_system; variable_storage)
            !isnothing(callback) && call!(callback, variable_storage)
            break
        end

        for i in 1:p
            # if we just updated, we can skip the first one
            if !(update_hamiltonian_at_each_step && i == 1) 
                update_hamiltonian!(gpe_system, i)
                Mϕi_valid = variable_storage.Mϕ_isValid[i]
                invalidate_cache!(variable_storage, i)
                # we only changed A, so Mϕ remains valid if it was before (but not Aϕ, R, etc.)
                variable_storage.Mϕ_isValid[i] = Mϕi_valid
            end
            gradient_function!(
                riemannian_gradient[i], ϕ[i], gpe_system, i, solver, variable_storage;
                gradient_kwargs...,
            )
            τ[i] = determine(
                step_size, ϕ, gpe_system, riemannian_gradient, i; 
                retraction!, variable_storage,
            )
            lmul!(-1 * τ[i], riemannian_gradient[i])
            retraction!(
                ϕ, riemannian_gradient, gpe_system, i;
                variable_storage, update_gpe_system = false,
            )
            update_weighted_mass_matrix!(gpe_system, ϕ, i, i)
        end
        update_hamiltonian_at_each_step && update_hamiltonian!(gpe_system)
        !isnothing(callback) && call!(callback, variable_storage)
    end

end

# Aliases

gradient_descent_L2(ϕ₀, gpe_system; kwargs...) =
    riemannian_gradient_descent(ϕ₀, gpe_system, gradient_L2!; kwargs...)

alternating_gradient_descent_L2(ϕ₀, gpe_system; kwargs...) =
    alternating_gradient_descent(ϕ₀, gpe_system, gradient_L2_componentwise!; kwargs...)

gradient_descent_L2_preconditioned(ϕ₀, gpe_system; kwargs...) =
    riemannian_gradient_descent(ϕ₀, gpe_system, gradient_L2_preconditioned!; kwargs...)

alternating_gradient_descent_L2_preconditioned(ϕ₀, gpe_system; kwargs...) =
    alternating_gradient_descent(
        ϕ₀, gpe_system, gradient_L2_preconditioned_componentwise!;
        kwargs...,
    )

gradient_descent_energy_adaptive(ϕ₀, gpe_system; kwargs...) =
    riemannian_gradient_descent(ϕ₀, gpe_system, gradient_energy_adaptive!; kwargs...)

alternating_gradient_descent_energy_adaptive(ϕ₀, gpe_system; kwargs...) =
    alternating_gradient_descent(ϕ₀, gpe_system, gradient_energy_adaptive_componentwise!; kwargs...)

gradient_descent_Lagrangian(ϕ₀, gpe_system; ω = 1.0, kwargs...) =
    riemannian_gradient_descent(
        ϕ₀, gpe_system, gradient_Lagrangian!; 
        ω, allocate_matrix = true, kwargs...,
    )

alternating_gradient_descent_Lagrangian(ϕ₀, gpe_system; ω = 1.0, kwargs...) =
    alternating_gradient_descent(
        ϕ₀, gpe_system, gradient_Lagrangian_componentwise!;
        ω, allocate_matrix = true, kwargs...,
    )

