abstract type StepSize end

struct ConstantStepSize <: StepSize
    τ::Number
end

function determine(step_size::ConstantStepSize, ϕ, gpe_system, gradient, i; kwargs...)
    return step_size.τ
end

reset!(step_size::ConstantStepSize) = nothing

