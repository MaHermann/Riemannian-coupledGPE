# Introduction

This is the code for the experiments in the paper **Riemannian Optimisation Methods for Ground States of Multicomponent Bose--Einstein Condensates**. The aim of the experiments is to find the ground state, i.e. the state with the minimal energy for systems of coupled Gross-Pitaevskii equations. For details, we refer to the paper.

# Usage

To reproduce the experiments, you need to run `experiments/optical_lattice.jl` for the 1D example (Section 8.1) and `experiments/localization.jl` for the 2D example (Section 8.2). The parameters can be set in the `@parameters` section in the beginning. Running these scripts will produce a file in the format `experiment_YYYY-MM-DD_HHMMSS` that contains the logging data as well as the final ground state and can be loaded by the `load_experiment` function. The best way to evaluate these results is via the files `experiments/evaluate_experiments_1D.jl` and `experiments/evaluate_experiments_2D.jl`, which can be used in a notebook-like style.

## Environment

You should run the experiments and evaluations in the provided package environment, i.e., first run `using Pkg; Pkg.activate("./Riemannian-coupledGPE/"); Pkg.instantiate();` inside julia, and when calling the files, use the `--project` flag, e.g. `julia --project=Riemannian-coupledGPE/ experiments/optical_lattice.jl`.

## Caveats

When using `Newton_method` as `optimization_algorithm`, you have to choose `:A0normNewton` as preconditioner, but only in this case! When used together with a gradient descent method (for which you have to use `:A0norm`), this will cause unpredictible behavior. This unfortunately has to do with missing size checks in `IncompleteLU.jl` and a disabled bounds check there.

Depending on your system, the 2D experiments might be too large too compute. In order to at least get the general flavour, you can try to run them with a lower n_elements (e.g. 2^6 and an l_ε of 5).

# Citation
When using this repository in your own work, please cite our paper:

    @article{altmann2024riemannian,
        title = {Riemannian optimisation methods for ground states of multicomponent Bose-Einstein condensates},
        author = {Altmann, Robert and Hermann, Martin and Peterseim, Daniel and Stykel, Tatjana},
        year = {2024},
        journal = {ArXiv e-print 2411.09617},
        number = {},
        doi = {10.48550/arXiv.2411.09617},
    }
