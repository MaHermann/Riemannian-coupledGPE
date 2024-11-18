abstract type Potential end

struct ConstantPotential{T} <: Potential
    potential_function
    weighted_mass_matrix::T
end

function ConstantPotential(potential_function, grid_context::GridContext)
    weighted_mass_matrix = assemble_weighted_mass_matrix(potential_function, grid_context)
    return ConstantPotential(potential_function, weighted_mass_matrix)
end

function add_potentials(potential1::ConstantPotential, potential2::ConstantPotential)
    return ConstantPotential(
        x -> (potential1.potential_function(x) + potential2.potential_function(x)),
        potential1.weighted_mass_matrix .+ potential2.weighted_mass_matrix,
    )
end

function matrix_representation(potential::ConstantPotential, _::Number)
    return potential.weighted_mass_matrix
end

function create_potential_function_from_points_1D(xs, values)  #assumes equal spacing
    xs = deepcopy(xs)
    values = deepcopy(values)
    a,b = xs[1], xs[end]
    ε = xs[2] - xs[1]
    function V(x)
        if x[1] <= a
            return values[1]
        elseif x[1] >= b
            return values[end]
        else
            return values[floor(Int, (x[1]-a)/ε) + 1]
        end
    end
    return V
end

function create_random_checkerboard_potential_2D(
    grid_context, l_ε; 
    p = 0.5, α = 0, β = 2^(2 * l_ε),
)
    segments = [rand() < p for _ in 1:2^l_ε, _ in 1:2^l_ε]
    ε = 2.0^(-l_ε)
    values = α .+ (β - α) .* segments
    function V(x)
        if x[1] == 0
            x[1] = ε/2
        end
        if x[2] == 0
            x[2] = ε/2
        end
        return values[ceil(Int, x[1]/ε), ceil(Int, x[2]/ε)]
    end
    return ConstantPotential(V, grid_context)
end

function create_periodic_potential_2D(
    grid_context, l_ε;
    α = 0, β = 2^(2 * l_ε), ε = 2.0^(-l_ε),
)
    return ConstantPotential(
        x ->  ceil(x[1]/ε)%2 == 0 && ceil(x[2]/ε)%2 != 0 ? α : β, grid_context,
    )
end

function rescale_potential_function(V, scale, shift)
    return x -> V((x[1] - shift)/scale)
end

function plot_potential_1D(interval, potential::ConstantPotential; 
    alpha = 0.1, scaling = 1, kwargs...
)
    V = x -> potential.potential_function(x)
    plot(interval, [scaling * V(x) for x in interval], fill = (0, alpha); kwargs...)
end

function plot_potential_1D!(interval, potential::ConstantPotential;
    alpha = 0.1, scaling = 1, kwargs...
)
    V = x -> potential.potential_function(x)
    plot!(interval, [scaling * V(x) for x in interval], fill = (0, alpha); kwargs...)
end

function plot_potential_2D(
    x_interval, y_interval, potential::ConstantPotential;
    scaling = 1, kwargs...
)
    V = x -> potential.potential_function(x)
    # flip x and y so the horizontal is x as is common in plotting
    heatmap(
        x_interval, y_interval, [scaling * V([x , y]) for y in y_interval, x in x_interval]; 
        kwargs...,
    )
end

function plot_potential_2D!(
    x_interval, y_interval, potential::ConstantPotential;
    scaling = 1, kwargs...
)
    V = x -> potential.potential_function(x)
    # flip x and y so the horizontal is x as is common in plotting
    heatmap!(
        x_interval, y_interval, [scaling * V([x , y]) for y in y_interval, x in x_interval]; 
        kwargs...,
    )
end
