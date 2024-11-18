abstract type Callback end

struct LoggingCallback{VI, V, VVI, VV, VS<:VariableStorage} <: Callback
    n_matrix_multiplications::VI
    energies::V
    n_inner_iterations::VVI
    solver_resnorms::VV
    τs::VV
    σs::VV
    residual_values::VV
    variable_storage::VS
end

function LoggingCallback(n, p, max_iter; T = Float64)
    variable_storage = VariableStorage(n, p, T)
    energies = zeros(max_iter)
    residual_values = ones(max_iter, p)
    solver_resnorms = zeros(max_iter, 2 * p)
    n_inner_iterations = zeros(Int, max_iter, 2 * p)
    n_matrix_multiplications = zeros(Int, max_iter)
    τs = zeros(max_iter, p)
    σs = zeros(max_iter, p)
    return LoggingCallback(
        n_matrix_multiplications, energies, n_inner_iterations,
        solver_resnorms, τs, σs, residual_values, variable_storage,
    )
end

function call!(callback::LoggingCallback, variable_storage)
    variables = variable_storage.variables
    n_step = variables["n_step"][1]
    ϕ = variables["ϕ"]
    gpe_system = variables["gpe_system"]
    callback.n_matrix_multiplications[n_step] =
        gpe_system.n_matrix_multiplications[1]
    current_energy = energy(
        ϕ, gpe_system;
        variable_storage = callback.variable_storage,
        update_gpe_system = false, count_multiplications = false,
    )
    callback.energies[n_step] = current_energy
    @views residuals!(
        callback.residual_values[n_step, :], ϕ, gpe_system;
        variable_storage =  callback.variable_storage,
        update_gpe_system = false, count_multiplications = false,
    )
    invalidate_cache!(callback.variable_storage)
    solver = variables["solver"]
    if !isempty(solver.history)
        for (i, history) in enumerate(solver.history)
            callback.n_inner_iterations[n_step, i] = history.iters
            if history.iters > 0
                callback.solver_resnorms[n_step, i] = history[:resnorm][end]
            end
        end
    end
    callback.τs[n_step,:] .= variables["τ"]
    callback.σs[n_step,:] .= variable_storage.σ
    # cleanup
    gpe_system.n_matrix_multiplications[1] = 0
    solver.history = ConvergenceHistory[]
end

function call!(callback::Function, variable_storage)
    callback(variable_storage)
end