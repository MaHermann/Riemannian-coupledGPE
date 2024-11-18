function energy(
    ϕ::PFrame{p}, gpe_system;
    variable_storage = VariableStorage(ϕ),
    update_gpe_system = true, count_multiplications = true,
) where {p}
    update_gpe_system && update!(gpe_system, ϕ)
    fixed_part = gpe_system.fixed_part
    Aϕ = variable_storage.Aϕ
    v = variable_storage.temp_n

    # faster than calling sum
    sum_over_p = 0
    for i in 1:p
        # energy =  0.5 * fixed part  + 0.25 * nonlinearity 
        #        = 0.25 * hamiltonian + 0.25 * fixed_part
        get_Aϕ!(variable_storage, ϕ[i], gpe_system, i; count_multiplications)
        sum_over_p += 0.25 * (ϕ[i] ⋅ Aϕ[i])
        count_multiplications && (gpe_system.n_matrix_multiplications[1] += 1)
        mul!(v, fixed_part[i], ϕ[i])
        sum_over_p += 0.25 * (ϕ[i] ⋅ v)
    end
    return sum_over_p
end

residuals(ϕ::PFrame{p}, gpe_system; kwargs...) where {p} = 
    residuals!(zeros(p), ϕ, gpe_system; kwargs...)

function residuals!(values, ϕ::PFrame{p}, gpe_system;
    variable_storage = VariableStorage(ϕ),
    update_gpe_system = true, count_multiplications = true,
) where {p}
    update_gpe_system && update!(gpe_system, ϕ)
    for i in 1:p
        values[i] = get_r!(
            variable_storage, ϕ[i], gpe_system, i;
            count_multiplications,
        )
    end
    return values
end

function random_normed_PFrame(gpe_system)
    n_components = size(gpe_system.interactions, 1)
    random_components = [
        randn(size(gpe_system.grid_context.M, 1)) 
        for _ in 1:n_components
    ]
    return normalized_PFrame(random_components, gpe_system)
end

function constant_normed_PFrame(gpe_system)
    n_components = size(gpe_system.interactions, 1)
    constant_components = [
        ones(size(gpe_system.grid_context.M, 1)) 
        for _ in 1:n_components
    ]
    return normalized_PFrame(constant_components, gpe_system)
end

function normalized_PFrame(components, gpe_system)
    return PFrame(SVector{size(components, 1)}(
        components[i] ./
            sqrt(components[i]' * gpe_system.grid_context.M * components[i] / 
                gpe_system.masses[i])
        for i in 1:size(components,1)
    ))
end

