@views function Newton_method(
    ϕ₀::PFrame{p}, gpe_system;
    ω = 1.0, kwargs...,
) where {p}
    arguments = Dict{Symbol, Any}(kwargs)

    n = size(ϕ₀[1], 1)

    retraction = get_retraction(arguments)
    iteration = get_iteration(arguments)
    termination_criterion = get_termination_criterion(arguments)
    callback = get_callback(arguments)
    step_size = get_step_size(arguments)
    variable_storage = VariableStorage(ϕ₀; gpe_system.grid_context)
    solver = get_solver(arguments)

    # preallocate and initialize
    ϕ = deepcopy(ϕ₀)
    direction = PFrame(SVector{p}(zero(ϕ₀[i]) for i in 1:p))
    b = zeros(n*p)
    u = zeros(n*p)
    temp_large = zeros(n*p)

    n_step = [0]                # array to have fixed memory location
    τ = zeros(p)
    Newton_residual = [0.0]

    Aϕ = variable_storage.Aϕ
    Mϕ = variable_storage.Mϕ
    σ  = variable_storage.σ
    temp = variable_storage.temp_n
    temp_G = variable_storage.G
    masses = gpe_system.masses
    M  = gpe_system.grid_context.M

    variable_storage.variables["b"] = b
    variable_storage.variables["u"] = u
    variable_storage.variables["ϕ"] = ϕ
    variable_storage.variables["n_step"] = n_step
    variable_storage.variables["gpe_system"] = gpe_system
    variable_storage.variables["τ"] = τ
    variable_storage.variables["solver"] = solver

    update!(gpe_system, ϕ; compute_offdiagonals = true)

    loop!(
        ϕ, ω, M, direction, b, u, temp_large, Aϕ, Mϕ, σ, temp, temp_G, masses, gpe_system,
        iteration, callback, termination_criterion, variable_storage, step_size,
        retraction, solver, n_step, τ,
    )
    return ϕ
end

# Function barrier for type stabillity
function loop!(
    ϕ::PFrame{p}, ω, M, direction::PFrame{p}, b, u, temp_large, Aϕ, Mϕ, σ, temp, temp_G,
    masses, gpe_system, iteration, callback, termination_criterion, variable_storage,
    step_size, retraction!, solver, n_step, τ,
) where {p}
    n = size(ϕ[1], 1)
    for i in iteration
        n_step[1] = i

        if is_met(termination_criterion, ϕ, gpe_system; variable_storage)
            !isnothing(callback) && call!(callback, variable_storage)
            break
        end

        for i in 1:p
            get_Aϕ!(variable_storage, ϕ[i], gpe_system, i)
            get_Mϕ!(variable_storage, ϕ[i], gpe_system, i)
            σ[i] = get_σ!(variable_storage, ϕ[i], gpe_system, i)
            temp_G[i] .= gpe_system.hamiltonian[i]
            # This can fail if the fixed part contains zeros where the densities do not,
            # for example for a periodic potential with 0 entries,
            # but this shouldn't be the case and this is much much faster
            temp_G[i].nzval .+= 2 .* gpe_system.interactions[i,i] .* 
                gpe_system.weighted_mass_matrices[i][i].nzval
            temp_G[i].nzval .-= ω .* σ[i].* M.nzval
        end

        function projected_h(y, x)
            temp_large .= x                          # don't modify x
            P_ϕϕᵀM_perp!(temp_large, ϕ, Mϕ, masses)
            h_ω!(y, temp_large, temp, temp_G, gpe_system)
            P_Mϕϕᵀ_perp!(y, ϕ, Mϕ, masses)
        end
        H = LinearMap(projected_h, n*p, issymmetric = true, ismutating = true)

        for j in 1:p
            J = (j-1)*n+1 : j*n
            b[J] .= get_R!(variable_storage, ϕ[j], gpe_system, j)
        end

        P_Mϕϕᵀ_perp!(b, ϕ, Mϕ, masses)
        u .= b

        # compute the relative tolerance in dependence of the residual
        abstol_old = solver.abstol
        reltol_old = solver.reltol
        r = sqrt(sum(get_r!(variable_storage, ϕ[i], gpe_system, i)^2 for i in 1:p))
        solver.abstol = r * solver.reltol
        solver.reltol = 0.0

        solve!(u, H, b, solver)
        
        solver.abstol = abstol_old
        solver.reltol = reltol_old

        P_ϕϕᵀM_perp!(u, ϕ, Mϕ, masses)

        for j in 1:p
            J = (j-1)*n+1 : j*n
            direction[j] .= u[J]
        end

        for i in 1:p
            τ[i] = determine(
                step_size, ϕ, gpe_system, direction, i; 
                retraction!, variable_storage,
            )
            lmul!(-1 * τ[i], direction[i])
        end

        retraction!(
            ϕ, direction, gpe_system;
            variable_storage, update_gpe_system = false,
        )
        update!(gpe_system, ϕ; compute_offdiagonals = true)
        !isnothing(callback) && call!(callback, variable_storage)
    end
end

@views function h_ω!(y, x, temp, temp_G, gpe_system; count_multiplications = true)
    n = size(temp, 1)
    p = size(gpe_system.interactions, 1)
    for j in 1:p
        J = (j-1)*n+1 : j*n
        gpe_system.n_matrix_multiplications[1] += 1
        mul!(y[J], temp_G[j], x[J])
        # y[J] .= y[J] .+ B_j,jj * y[jj]
        for jj in 1:p
            if jj != j
                JJ = (jj-1)*n+1 : jj*n
                # here the factor 2 is already included
                mul_B!(temp, x[JJ], gpe_system, j, jj; count_multiplications)
                axpy!(1, temp, y[J])
            end
        end
    end
end

@views function P_ϕϕᵀM_perp!(
    u, ϕ::PFrame{p}, Mϕ::PFrame{p}, masses,
) where {p}
    n = size(ϕ[1], 1)
    for j in 1:p
        J = (j-1)*n+1 : j*n
        k = Mϕ[j] ⋅ u[J] / masses[j]
        axpy!(-1 * k, ϕ[j], u[J])
    end
end

@views function P_Mϕϕᵀ_perp!(
    u, ϕ::PFrame{p}, Mϕ::PFrame{p}, masses,
) where {p}
    n = size(ϕ[1], 1)
    for j in 1:p
        J = (j-1)*n+1 : j*n
        k = ϕ[j] ⋅ u[J] / masses[j]
        axpy!(-1 * k, Mϕ[j], u[J])
    end
end

