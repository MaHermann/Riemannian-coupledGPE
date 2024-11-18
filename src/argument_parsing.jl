DEFAULT_RETRACTION = normalization_retraction!
DEFAULT_MAX_ITER = 100
DEFAULT_STEP_SIZE = ConstantStepSize(1)
DEFAULT_SOLVER = CGSolver(Float64)

function get_default_solver()
    return DEFAULT_SOLVER
end

function get_retraction(arguments)
    if :retraction in keys(arguments)
        return arguments[:retraction]
    else
        return DEFAULT_RETRACTION
    end
end

function get_callback(arguments)
    if :callback in keys(arguments)
        return arguments[:callback]
    else
        return nothing
    end
end

function get_iteration(arguments)
    if :verbose in keys(arguments)
        verbose = arguments[:verbose]
    else
        verbose = false
    end
    if :max_iter in keys(arguments)
        max_iter = arguments[:max_iter]
    else
        max_iter = DEFAULT_MAX_ITER
    end
    if verbose
        return ProgressBar(1:max_iter)
    else
        return 1:max_iter
    end
end

function get_termination_criterion(arguments)
    if :termination_criterion in keys(arguments)
        return arguments[:termination_criterion]
    else
        return NoCriterion()
    end
end

function get_solver(arguments)
    if :solver in keys(arguments)
        return arguments[:solver]
    else
        return get_default_solver()
    end
end

function get_step_size(arguments)
    if :step_size in keys(arguments)
        step_size = arguments[:step_size]
    else
        step_size = DEFAULT_STEP_SIZE
    end
    if typeof(step_size) <: Number
        step_size = ConstantStepSize(step_size)
    end
    return step_size
end

function get_gradient_kwargs(arguments)
    gradient_kwargs = Dict()
    if :warmstart in keys(arguments)
        gradient_kwargs[:warmstart] = arguments[:warmstart]
    end
    if :ω in keys(arguments)
        gradient_kwargs[:ω] = arguments[:ω]
    end
    if :reltol_from_R in keys(arguments)
        gradient_kwargs[:reltol_from_R] = arguments[:reltol_from_R]
    end
    return gradient_kwargs
end