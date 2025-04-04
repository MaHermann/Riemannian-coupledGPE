"""
    GPESystem{T, dim, refshape, G, CSC_T, CI, V, VV, M, V_T}

    This manages the information and state of the GPE that is used. This includes
    the concrete potential, masses and nonlinearities, but also the Finite Element
    grid and the mass matrices of the current ϕ. It also takes care of counting 
    multiplications for performance comparisons.
    
    To interact with this, the preferred interface is via the `update!` function,
    which should be called after changing the current ϕ, and the three functions
    `mul_hamiltonian!`, `mul_M!` and `mul_B!`, that carry out in-place multiplications
    with the hamiltonian, the mass matrix and the second order matrices B_ij, 
    respectively.
"""

struct GPESystem{T, dim, refshape, G, CSC_T<:SparseMatrixCSC{T, Int64}, CI,
                V<:Vector{CSC_T}, VV<:Vector{V}, M<:Matrix{T}, V_T<:Vector{T}}
    L::CSC_T
    potential::Potential
    fixed_part::V
    hamiltonian::V
    weighted_mass_matrices::VV
    interactions::M
    masses::V_T
    grid_context::GridContext{T, dim, refshape, G, CSC_T, CI}
    Ne::M
    n_matrix_multiplications::Vector{Int}
end

function GPESystem(L, potential, interactions, masses, grid_context) 
    return GPESystem(
        L, potential,
        [L + matrix_representation(potential, i) for i in 1:size(masses, 1)],
        [allocate_matrix(grid_context.dof_handler) for i in 1:size(masses, 1)],
        [
            [allocate_matrix(grid_context.dof_handler) for _ in 1:i]
            for i in 1:size(masses, 1)
        ],
        interactions, masses, grid_context, 
        zeros(
            getnbasefunctions(grid_context.cellvalues),
            getnbasefunctions(grid_context.cellvalues)
        ), [0],
        )
end

function update!(gpe_system, ϕ; compute_offdiagonals = false)
    update_weighted_mass_matrices!(gpe_system, ϕ; compute_offdiagonals)
    update_hamiltonian!(gpe_system)
end

function update_weighted_mass_matrices!(
    gpe_system, ϕ::PFrame{p};
    compute_offdiagonals = false,
) where {p}
    if compute_offdiagonals
        for i in 1:p
            for j in 1:i
                update_weighted_mass_matrix!(gpe_system, ϕ, i, j)
            end
        end
    else
        for i in 1:p
            update_weighted_mass_matrix!(gpe_system, ϕ, i, i)
        end
    end
end

function update_weighted_mass_matrix!(gpe_system, ϕ, i, j) 
    assemble_density_weighted_mass_matrix!(
        gpe_system.weighted_mass_matrices[i][j], ϕ[i], ϕ[j], 
        gpe_system.grid_context, gpe_system.Ne,
    )
end

function update_hamiltonian!(gpe_system)
    p = size(gpe_system.masses, 1)
    for i in 1:p
        update_hamiltonian!(gpe_system, i)
    end
end

function update_hamiltonian!(gpe_system, i)
    p = size(gpe_system.masses, 1)
    gpe_system.hamiltonian[i] .= gpe_system.fixed_part[i]
    # This can fail if the fixed part contains zeros where the densities do not,
    # for example for a periodic potential with 0 entries,
    # but this shouldn't be the case and this is much much faster
    for j in 1:p
        gpe_system.hamiltonian[i].nzval .+= gpe_system.interactions[j,i] .*
            gpe_system.weighted_mass_matrices[j][j].nzval
    end
end

function mul_hamiltonian!(v, u, gpe_system, i; count_multiplications = true)
    count_multiplications && (gpe_system.n_matrix_multiplications[1] += 1)
    mul!(v, gpe_system.hamiltonian[i], u)
    return v
end

function mul_B!(
    v, u, gpe_system, i, j;
    count_multiplications = true,
)
    count_multiplications && (gpe_system.n_matrix_multiplications[1] += 1)
    if j > i                # symmetry / commutativity
        i, j = j, i
    end
    mul!(v, gpe_system.weighted_mass_matrices[i][j], u)
    lmul!(gpe_system.interactions[i,j] + gpe_system.interactions[j,i], v)
    return v
end

function mul_M!(v, u, gpe_system; count_multiplications = true)
    count_multiplications && (gpe_system.n_matrix_multiplications[1] += 1)
    mul!(v, gpe_system.grid_context.M, u)
    return v
end
