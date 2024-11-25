using Plots
using Statistics

include("../src/Riemannian-coupledGPE.jl")
using .MulticomponentBEC

## Physical parameters
dimension               = 1
domain                  = [-16.0, 16.0]
## Numerical parameters
n_elements              = 2^10
interp_degree           = 2
quad_degree             = 5
##
grid_context = generate_grid_context(
    dimension, n_elements, interp_degree, quad_degree;
    left = domain[1], right = domain[2],
)
##
results_folder = "1D_results" #change to where your "experiment_YYYY-MM-DD_HHMMSS" files are located
experiments = [
    load_experiment(results_folder*"/"*filename)
    for filename in readdir(results_folder)
];
##
for experiment in experiments
    filename, experiment = experiment
    experiment.parameters["filename"] = split(filename, "/")[end]
end
experiments = [experiment for (_, experiment) in experiments]
## sanity check
β = 10
plots = []
for experiment in experiments
    if experiment.parameters["β"] == β
        individual_plot = plot_PFrame(experiment.value, grid_context)
        individual_plot = plot!(
            title = string(experiment.parameters["optimization_algorithm"])
        )
        push!(plots, individual_plot)
        potential = ConstantPotential(
            experiment.parameters["potential_function"], grid_context,
        )
        gpe_system = GPESystem(
            assemble_stiffness_matrix(grid_context),
            potential, experiment.parameters["interactions"],
            experiment.parameters["masses"], grid_context,
        )
        println(
            "$(sqrt.(sum(experiment.log["initialization_residuals"].^2)))\t" *
            "$(energy(experiment.value, gpe_system))\t" *
            "$(sqrt(sum(residuals(experiment.value, gpe_system).^2)))\t" *
            "$(experiment.parameters["filename"])"
        )
    end
end
main_plot = plot(xlims = (-16, 16), legend = false)
for experiment in experiments 
    if experiment.parameters["β"] == β 
        main_plot = plot_PFrame!(experiment.value, grid_context)
    end
end
plot(main_plot, plots..., size = (2000, 2000))
##
β = 10
for experiment in experiments
    if experiment.parameters["β"] == 10
        n_iter = findfirst(
            sqrt.(sum(experiment.log["residual_values"].^2, dims = 2)) .< 1e-8
        )
        if isnothing(n_iter)
            n_iter = size(experiment.log["residual_values"], 1) - 2
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
        elapsed_time = experiment.metadata["elapsed_time"]
        println("$n_iter\t$(avg_matrix_mul)\t$(elapsed_time)\t$(filename)")
    end
end
