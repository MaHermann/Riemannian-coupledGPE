"""
    VariableStorage{p, T, P, G, V}

    This serves three purposes:
        - It preallocates memory to be used by performance critical computations,
        such as gradients etc.
        - It manages rudimentary caching for these computations
        - Finally, `variables` is a dictionary to store all sorts of information 
        that should be tracked and accessed e.g. by callbacks (such as the current
        iteration number, step sizes, ...)
        
    `Aϕ` and `Mϕ` are `PFrame`s storing the products of the Hamiltonian and
    mass matrix with the current point `ϕ`, `R` are the residual vectors for each
    component, `MR` their products with the mass matrix, `r` the square root of 
    the sum of their norms and `σ` the Rayleigh coefficients. The getters for these
    decide whether to use the current cached value or recompute them if necessary.
    `G` is a vector of optional sparse matrices that is used in the Lagrangian 
    gradient, and `temp_n` and `temp2_n` are arrays of size `n` that can be used 
    for temporary memory e.g. to save intermediate results, but for which no further 
    guarantees  are assumed (i.e. it can be overwritten in any function without 
    consequences).
"""
struct VariableStorage{p, T, P<:PFrame{p, T}, G, V<:AbstractVector{T}}
    Aϕ::P
    Mϕ::P
    R::P
    MR::P
    r::Vector{T}
    σ::Vector{T}
    Aϕ_isValid::Vector{Bool}
    Mϕ_isValid::Vector{Bool}
    R_isValid::Vector{Bool}
    MR_isValid::Vector{Bool}
    r_isValid::Vector{Bool}
    σ_isValid::Vector{Bool}
    G::G
    temp_n::V
    temp2_n::V
    variables::Dict{String, Any}
end

function VariableStorage(n, p, T; grid_context = nothing)
    return VariableStorage(
        PFrame(SVector{p}(zeros(T, n) for _ in 1:p)),
        PFrame(SVector{p}(zeros(T, n) for _ in 1:p)),
        PFrame(SVector{p}(zeros(T, n) for _ in 1:p)),
        PFrame(SVector{p}(zeros(T, n) for _ in 1:p)),
        [zero(T) for _ in 1:p],
        [zero(T) for _ in 1:p],
        [false for _ in 1:p], [false for _ in 1:p],
        [false for _ in 1:p], [false for _ in 1:p],
        [false for _ in 1:p], [false for _ in 1:p],
        isnothing(grid_context) ? nothing :
            [allocate_matrix(grid_context.dof_handler) for i in 1:p],
        zeros(T, n), zeros(T, n), Dict{String, Any}(),
)
end

function invalidate_cache!(variable_storage, i)
    variable_storage.Aϕ_isValid[i] = false
    variable_storage.Mϕ_isValid[i] = false
    variable_storage.R_isValid[i] = false
    variable_storage.MR_isValid[i] = false
    variable_storage.r_isValid[i] = false
    variable_storage.σ_isValid[i] = false
end

function invalidate_cache!(variable_storage::VariableStorage{p}) where {p}
    for i in 1:p
        invalidate_cache!(variable_storage, i)
    end
end

function get_Aϕ!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications = true)
    if !variable_storage.Aϕ_isValid[i]
        mul_hamiltonian!(
            variable_storage.Aϕ[i], ϕᵢ, gpe_system, i; 
            count_multiplications,
        )
        variable_storage.Aϕ_isValid[i] = true
    end
    return variable_storage.Aϕ[i]
end

function get_Mϕ!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications = true)
    if !variable_storage.Mϕ_isValid[i]
        mul_M!(variable_storage.Mϕ[i], ϕᵢ, gpe_system; count_multiplications)
        variable_storage.Mϕ_isValid[i] = true
    end
    return variable_storage.Mϕ[i]
end

function get_R!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications = true)
    if !variable_storage.R_isValid[i]
        Mϕ = get_Mϕ!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications)
        Aϕ = get_Aϕ!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications)
        σ  = get_σ!( variable_storage, ϕᵢ, gpe_system, i; count_multiplications)
        variable_storage.R[i] .= Aϕ .- (σ .* Mϕ)
        variable_storage.R_isValid[i] = true
    end
    return variable_storage.R[i]
end

function get_MR!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications = true)
    if !variable_storage.MR_isValid[i]
        R = get_R!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications)
        mul_M!(variable_storage.MR[i], R, gpe_system; count_multiplications)
        variable_storage.MR_isValid[i] = true
    end
    return variable_storage.MR[i]
end

function get_r!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications = true)
    if !variable_storage.r_isValid[i]
        R  = get_R!( variable_storage, ϕᵢ, gpe_system, i; count_multiplications)
        MR = get_MR!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications)
        variable_storage.r[i] = sqrt(R ⋅ MR)
        variable_storage.r_isValid[i] = true
    end
    return variable_storage.r[i]
end

function get_σ!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications = true)
    if !variable_storage.σ_isValid[i]
        Aϕ = get_Aϕ!(variable_storage, ϕᵢ, gpe_system, i; count_multiplications)
        variable_storage.σ[i] = (ϕᵢ ⋅ Aϕ) / gpe_system.masses[i]
        variable_storage.σ_isValid[i] = true
    end
    return variable_storage.σ[i]
end

VariableStorage(ϕ::PFrame{p}; grid_context = nothing) where {p} = 
    VariableStorage(size(ϕ[1], 1), p, eltype(ϕ[1]); grid_context)

