abstract type TerminationCriterion end

struct NoCriterion <: TerminationCriterion end
function is_met(criterion::NoCriterion, ϕ, gpe_system; kwargs...)
    return false
end

struct AbsoluteCriterion <: TerminationCriterion
    quantityOfInterest
    threshold::Number
end

function is_met(criterion::AbsoluteCriterion, ϕ, gpe_system; kwargs...)
    value = criterion.quantityOfInterest(ϕ, gpe_system; kwargs...)
    return value < criterion.threshold
end

struct SumResidual <: TerminationCriterion
    threshold::Number
    update_gpe_system::Bool
    count_multiplications::Bool
    temp_array::Vector{Float64}
end

function is_met(
    criterion::SumResidual, ϕ, gpe_system;
    variable_storage = VariableStorage(ϕ)
)
    value = sqrt.(sum(residuals!(
        criterion.temp_array, ϕ, gpe_system;
        variable_storage = variable_storage, 
        update_gpe_system = criterion.update_gpe_system,
        count_multiplications = criterion.count_multiplications,
    ).^2))
    return value < criterion.threshold
end
