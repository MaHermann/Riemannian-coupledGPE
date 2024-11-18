function gradient!(
    gradient::PFrame{p}, ϕ::PFrame{p}, gpe_system, gradient_componentwise!;
    solver = nothing, variable_storage = nothing, update_gpe_system = true,
    kwargs...
) where {p}
    isnothing(solver) && (solver = get_default_solver())
    isnothing(variable_storage) && (variable_storage = VariableStorage(ϕ; gpe_system.grid_context))
    update_gpe_system && update!(gpe_system, ϕ)
    for i in 1:p
        gradient[i] .= gradient_componentwise!(
            gradient[i], ϕ[i], gpe_system, i, solver, variable_storage;
            kwargs...
        )
    end
    return gradient
end

# Aliases
function gradient_L2!(
    gradient, ϕ, gpe_system;
    solver = nothing, variable_storage = nothing, 
    update_gpe_system = true, kwargs...,
)
    gradient!(
        gradient, ϕ, gpe_system, gradient_L2_componentwise!;
        solver, variable_storage, update_gpe_system, kwargs...
    )
end

function gradient_L2(
    ϕ::PFrame{p}, gpe_system;
    solver = nothing, variable_storage = nothing,
    update_gpe_system = true, kwargs...,
) where {p} 
    gradient_L2!(
        PFrame(SVector{p}(zeros(size(ϕ[i], 1)) for i in 1:p)), ϕ, gpe_system;
        solver, variable_storage, update_gpe_system, kwargs...,
    )
end

function gradient_L2_preconditioned!(
    gradient, ϕ, gpe_system;
    solver = nothing, variable_storage = nothing, 
    update_gpe_system = true, kwargs...,
)
    gradient!(
        gradient, ϕ, gpe_system, gradient_L2_preconditioned_componentwise!;
        solver, variable_storage, update_gpe_system, kwargs...,
    )
end

function gradient_L2_preconditioned(
    ϕ::PFrame{p}, gpe_system;
    solver = nothing, variable_storage = nothing,
    update_gpe_system = true, kwargs...,
) where {p}
    gradient_L2_preconditioned!(
        PFrame(SVector{p}(zeros(size(ϕ[i], 1)) for i in 1:p)), ϕ, gpe_system;
        solver, variable_storage, update_gpe_system, kwargs...,
    )
end

function gradient_energy_adaptive!(
    gradient, ϕ, gpe_system;
    solver = nothing, variable_storage = nothing,
    update_gpe_system = true, kwargs...,
)
    gradient!(
        gradient, ϕ, gpe_system, gradient_energy_adaptive_componentwise!;
        solver, variable_storage, update_gpe_system, kwargs...,
    )
end

function gradient_energy_adaptive(
    ϕ::PFrame{p}, gpe_system;
    solver = nothing, variable_storage = nothing,
    update_gpe_system = true, kwargs...,
) where {p}
    gradient_energy_adaptive!(
        PFrame(SVector{p}(zeros(size(ϕ[i], 1)) for i in 1:p)), ϕ, gpe_system;
        solver, variable_storage, update_gpe_system, kwargs...,
    )
end

function gradient_Lagrangian!(
    gradient, ϕ, gpe_system;
    solver = nothing, variable_storage = nothing, 
    update_gpe_system = true, ω = 1.0, kwargs...,
)
    gradient!(
        gradient, ϕ, gpe_system, gradient_Lagrangian_componentwise!;
        solver, variable_storage, update_gpe_system, ω, kwargs...,
    )
end

function gradient_Lagrangian(
    ϕ::PFrame{p}, gpe_system;
    solver = nothing, variable_storage = nothing,
    update_gpe_system = true, ω = 1.0, kwargs...,
) where {p}
    gradient_Lagrangian!(
        PFrame(SVector{p}(zeros(size(ϕ[i], 1)) for i in 1:p)), ϕ, gpe_system;
        solver, variable_storage, update_gpe_system, ω, kwargs...,
    )
end

# Componentwise gradients
function gradient_L2_componentwise!(
    u, ϕᵢ, gpe_system, i, solver, variable_storage;
    kwargs...,
)
    n = size(ϕᵢ, 1)
    m(y, x) =  mul_M!(y, x, gpe_system)
    M = LinearMap(m, n, issymmetric = true, ismutating = true)

    Aϕ = get_Aϕ!(variable_storage, ϕᵢ, gpe_system, i)
    σ  = get_σ!( variable_storage, ϕᵢ, gpe_system, i)

    #warmstart
    u .= σ .* ϕᵢ

    solve!(u, M, Aϕ, solver)

    axpy!(-1 * σ, ϕᵢ, u)
    return u
end

function gradient_L2_preconditioned_componentwise!(
    u, ϕᵢ, gpe_system, i, solver, variable_storage;
    kwargs...,
)
    n = size(ϕᵢ, 1)
    a(y, x) = mul_hamiltonian!(y, x, gpe_system, i)
    A = LinearMap(a, n, issymmetric = true, ismutating = true)

    Mϕ = get_Mϕ!(variable_storage, ϕᵢ, gpe_system, i)
    σ  = get_σ!( variable_storage, ϕᵢ, gpe_system, i)
    massᵢ = gpe_system.masses[i]

    #warmstart
    u .= σ .* ϕᵢ

    solve!(u, A, Mϕ, solver)

    σ_A = (u ⋅ Mϕ) / massᵢ
    axpy!(-σ_A, ϕᵢ, u)

    lmul!(-σ, u)
    return u
end

function gradient_energy_adaptive_componentwise!(
    u, ϕᵢ, gpe_system, i, solver, variable_storage;
    kwargs...,
)
    n = size(ϕᵢ, 1)
    a(y, x) = mul_hamiltonian!(y, x, gpe_system, i)
    A = LinearMap(a, n, issymmetric = true, ismutating = true)

    Mϕ = get_Mϕ!(variable_storage, ϕᵢ, gpe_system, i)
    R  = get_R!( variable_storage, ϕᵢ, gpe_system, i)
    massᵢ = gpe_system.masses[i]

    u .= R
    # compute the relative tolerance in dependence of the residual
    abstol_old = solver.abstol
    reltol_old = solver.reltol
    r = get_r!(variable_storage, ϕᵢ, gpe_system, i)
    solver.abstol = r * solver.reltol
    solver.reltol = 0.0

    solve!(u, A, R, solver)

    solver.abstol = abstol_old 
    solver.reltol = reltol_old

    σ_R = (Mϕ ⋅ u) / massᵢ
    σ_R = 1 / (1 - σ_R)
    
    axpy!(-1, ϕᵢ, u)
    lmul!(-σ_R, u)

    axpy!(-1, ϕᵢ, u)
    lmul!(-1, u)
    return u
end

# We have to be careful here, as we use CG as a solver, and we have no guarantee
# that G is positive definite at a random point (but it should be if it is close
# enough to a critical point)
function gradient_Lagrangian_componentwise!(
    u, ϕᵢ, gpe_system, i, solver, variable_storage;
    ω = 1.0, kwargs...,
)
    Mϕ = get_Mϕ!(variable_storage, ϕᵢ, gpe_system, i)
    R  = get_R!( variable_storage, ϕᵢ, gpe_system, i)
    σ  = get_σ!( variable_storage, ϕᵢ, gpe_system, i)
    M = gpe_system.grid_context.M

    G = variable_storage.G[i]
    G .= gpe_system.hamiltonian[i]
    # This can fail if the fixed part contains zeros where the densities do not,
    # for example for a periodic potential with 0 entries,
    # but this shouldn't be the case and this is much much faster
    G.nzval .+= 2 .* gpe_system.interactions[i,i] .* 
        gpe_system.weighted_mass_matrices[i][i].nzval
    G.nzval .-= ω .* σ.* M.nzval

    function g_ω(y, x)
        gpe_system.n_matrix_multiplications[1] += 1
        mul!(y, G, x)
        return y
    end
    G_ω = LinearMap(g_ω, size(u, 1), issymmetric = true, ismutating = true)

    u .= R
    # compute the relative tolerance in dependence of the residual
    abstol_old = solver.abstol
    reltol_old = solver.reltol
    r = get_r!(variable_storage, ϕᵢ, gpe_system, i)
    solver.abstol = r * solver.reltol
    solver.reltol = 0.0

    solve!(u, G_ω, R, solver)

    # renaming for clarity
    v = R

    v .= Mϕ
    solve!(v, G_ω, Mϕ, solver)

    solver.abstol = abstol_old 
    solver.reltol = reltol_old

    σ_G = (Mϕ ⋅ u) / (Mϕ ⋅ v)
    axpy!(-σ_G, v, u)
    return u
end

