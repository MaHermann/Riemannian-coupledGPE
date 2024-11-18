module MulticomponentBEC

using Ferrite, SparseArrays
using LinearAlgebra
using IncompleteLU
using IterativeSolvers, LinearMaps
using Plots
using ProgressBars
using Serialization
using StaticArrays

import Dates: format, now

export CGSolver, get_A0preconditioner, get_identitypreconditioner

export Experiment, load_experiment, save_experiment, @parameters, @experiment

export assemble_stiffness_matrix, GridContext, generate_grid_context, grid_points

export VariableStorage

export normalization_retraction!

export gradient_descent_L2, gradient_descent_L2_preconditioned,
        gradient_descent_energy_adaptive, gradient_descent_Lagrangian,
        alternating_gradient_descent_L2,
        alternating_gradient_descent_L2_preconditioned,
        alternating_gradient_descent_energy_adaptive,
        alternating_gradient_descent_Lagrangian

export Newton_method

export GPESystem

export AbsoluteCriterion, SumResidual

export gradient_L2, gradient_energy_adaptive, gradient_L2_preconditioned,
        gradient_Lagrangian

export ConstantStepSize, reset!

export PFrame, inner, density, plot_PFrame, plot_PFrame!

export ConstantPotential, rescale_potential, create_periodic_potential_2D, add_potentials,
        create_random_checkerboard_potential_2D, plot_potential_1D, plot_potential_1D!,
        plot_potential_2D, plot_potential_2D!

export energy, residuals, residuals!, random_normed_PFrame, constant_normed_PFrame

export LoggingCallback

include("fem_util.jl")
include("solver_util.jl")
include("pframe.jl")
include("variable_storage.jl")
include("retractions.jl")
include("potential.jl")
include("gpe_system.jl")
include("step_size.jl")
include("termination_criterion.jl")
include("gradients.jl")
include("argument_parsing.jl")
include("gradient_methods.jl")
include("second_order_methods.jl")
include("experiment_util.jl")
include("util.jl")
include("callback.jl")

end
