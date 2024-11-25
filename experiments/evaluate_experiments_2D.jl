using Plots
using Statistics

include("../src/Riemannian-coupledGPE.jl")
using .MulticomponentBEC

## Physical parameters
dimension               = 2
domain                  = [0.0, 1.0]
## Numerical parameters
n_elements              = 2^10
interp_degree           = 2
quad_degree             = 5
l_ε                     = 6
##
grid_context = generate_grid_context(
    dimension, n_elements, interp_degree, quad_degree;
    left = domain[1], right = domain[2],
);
##
results_folder = "2D_results" #change to where your "experiment_YYYY-MM-DD_HHMMSS" files are located
experiments = [
    load_experiment(results_folder*"/"*filename)
    for filename in readdir(results_folder)
]
##
for experiment in experiments
    filename, experiment = experiment
    experiment.parameters["filename"] = split(filename, "/")[end]
end
experiments = [experiment for (_, experiment) in experiments]
## sanity check
potential_type = :periodic
plots = []
for experiment in experiments
    if experiment.parameters["potential_type"] == potential_type
        state_plot = plot_PFrame(
            experiment.value, grid_context;
            plot_sum = false, n_points = 50,
        )
        density_plot = plot_PFrame(
            MulticomponentBEC.density(experiment.value), grid_context;
            plot_sum = false, n_points = 50,
        )
        individual_plot = plot(state_plot, density_plot, layout = (2,1))
        push!(plots, individual_plot)
        n_iter = findfirst(
            sqrt.(sum(experiment.log["residual_values"].^2, dims = 2)) .< 1e-8
        )
        if isnothing(n_iter)
            n_iter = size(experiment.log["residual_values"], 1)
        else
            n_iter = n_iter[1]
        end
        println(
            "$(sqrt.(sum(experiment.log["initialization_residuals"].^2)))\t" *
            "$(experiment.log["energies"][n_iter])\t" * 
            "$(sqrt(sum(experiment.log["residual_values"][n_iter,:].^2)))\t" * 
            "$(experiment.parameters["filename"])"
        )
    end
end
plot(plots..., size=(1000, 800))
##
potential_type = :periodic
for experiment in experiments
    if experiment.parameters["potential_type"] == potential_type
        n_iter = findfirst(
            sqrt.(sum(experiment.log["residual_values"].^2, dims = 2)) .< 1e-8
        )
        if isnothing(n_iter)
            continue
        else
            n_iter = n_iter[1]
        end
        avg_matrix_mul = round(mean(
            experiment.log["n_matrix_multiplications"][1:n_iter]),
            digits = 1,
        )
        n_inner_iterations = round(mean(
            sum(
                experiment.log["n_inner_iterations"],
                dims = 2,
            )[1:n_iter]),
            digits = 1,
        )
        filename = experiment.parameters["filename"]
        time = experiment.metadata["elapsed_time"]
        println("$n_iter\t$(avg_matrix_mul)\t$(time)\t$(filename)")
    end
end

