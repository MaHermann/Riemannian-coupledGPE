abstract type Solver end

"""
    Mainly a thin wrapper around `IterativeSolvers.jl`s cg!

    Originally intended to be more flexible and e.g. also allow 
    functions from LinearSolve.jl which have a slightly different 
    interace, but for various reasons this turned out to be not 
    necessary.
"""
mutable struct CGSolver{P,R,A,M,S} <: Solver
    preconditioner::P
    isLogging::Bool
    abstol::R
    reltol::A
    maxiter::M
    history::Vector{ConvergenceHistory}
    statevariables::S
end

# Set all undefined values to the default values of Iterativesolvers.cg!
# This is not particularly clean as we may break on changes in IterativeSolvers.jl
# but its avoids a lot of dispatch issues
function CGSolver(
    T;
    preconditioner = Identity(), isLogging = false, 
    abstol = zero(real(T)), reltol = sqrt(eps(real(T))),
    maxiter = -1,
)
    isnothing(abstol) && (abstol = zero(real(T)))
    isnothing(reltol) && (reltol = sqrt(eps(real(T))))
    isnothing(maxiter) && (maxiter = -1)
    return CGSolver(
        preconditioner, isLogging, abstol, reltol, maxiter, 
        ConvergenceHistory[], CGStateVariables(zeros(0), zeros(0), zeros(0)),
    )
end

function solve!(u, A, b, solver::CGSolver)
    # slight optimization to reuse statevars
    if size(u, 1) != size(solver.statevariables.u, 1)
        solver.statevariables = CGStateVariables(zero(u), similar(u), similar(u))
    else
        solver.statevariables.u .= 0
        solver.statevariables.r .= 0
        solver.statevariables.c .= 0
    end
    if solver.maxiter < 0
        maxiter = size(A, 2)
    else
        maxiter = solver.maxiter
    end

    if solver.isLogging
        u, ch = cg!(
            u, A, b; 
            log = true, abstol = solver.abstol, reltol = solver.reltol,
            Pl = solver.preconditioner, maxiter = maxiter,
            statevars = solver.statevariables,
        )
        push!(solver.history, ch)
    else
        u = cg!(
            u, A, b; 
            log = false, abstol = solver.abstol, reltol = solver.reltol,
            Pl = solver.preconditioner, maxiter = maxiter,
            statevars = solver.statevariables,
        )
        end
    return u
end

# assumes ConstantPotential, i.e. the same in all components
function get_A0preconditioner(gpe_system; τ = 0.01, isNewton = false)
    if isNewton
        # be extremely careful with this, as ilu does not do dimension checks AND
        # even actively disables them internally, leading to possible silent fails
        p = size(gpe_system.interactions, 1)
        preconditioner = ilu(
            hcat(
                (vcat(
                    (allocate_matrix(
                        gpe_system.grid_context.dof_handler,
                        ) for _ in 1:(i-1))...,
                    gpe_system.fixed_part[1],
                    (allocate_matrix(
                        gpe_system.grid_context.dof_handler
                        ) for _ in (i+1):p)...,
                )  for i in 1:p)...
            ),
            τ = τ,
        )
    else
        preconditioner = ilu(gpe_system.fixed_part[1], τ = τ)
    end
    return preconditioner
end

get_identitypreconditioner() = Identity()

