function normalization_retraction!(
    ϕ::PFrame{p}, z::PFrame{p}, gpe_system;
    variable_storage::VariableStorage = VariableStorage(ϕ),
    update_gpe_system = true, count_multiplications = true,
) where {p}
    update_gpe_system && update!(gpe_system, ϕ)
    for i in 1:p
        normalization_retraction!(
            ϕ, z, gpe_system, i; 
            variable_storage, count_multiplications, 
            update_gpe_system = false,
        )
    end
    return ϕ
end

function normalization_retraction!(
    ϕ::PFrame{p}, z::PFrame{p}, gpe_system, i; 
    variable_storage::VariableStorage = VariableStorage(ϕ),
    update_gpe_system = true, count_multiplications = true,
) where {p}
    update_gpe_system && update!(gpe_system, ϕ)
    Mϕ = variable_storage.Mϕ

    ϕ[i] .= ϕ[i] .+ z[i]
    invalidate_cache!(variable_storage, i)
    get_Mϕ!(variable_storage, ϕ[i], gpe_system, i; count_multiplications)
    k = sqrt(gpe_system.masses[i] / (ϕ[i] ⋅ Mϕ[i]))
    lmul!(k, ϕ[i])
    lmul!(k, variable_storage.Mϕ[i])
    return ϕ
end
