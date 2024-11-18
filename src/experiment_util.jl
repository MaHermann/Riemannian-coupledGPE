struct Experiment
    value
    parameters::Dict
    log::Dict
    metadata::Dict
end

function Experiment(value, parameters, log, comment = nothing)
    metadata = Dict(("date" => now()), "nthreads" => Threads.nthreads(),
                "version" => VERSION, "ENV" => ENV, "cpu_info" => Sys.cpu_info(),
                "total_memory" => Sys.total_memory() / 2^20)
    if !isnothing(comment)
        metadata["comment"] = comment
    end
    return Experiment(value, parameters, log, metadata)
end

function save_experiment(filename::String, experiment::Experiment)
    serialize(filename, experiment)
end

function save_experiment(experiment::Experiment)
    save_experiment("experiment_"*format(now(), "yyyy-mm-dd_HHMMSS"), experiment)
end

function load_experiment(filename::String)
    return (filename, deserialize(filename))
end

macro parameters(dict, expression)
    assignments = []
    dictionary_assignments = []
    if expression.head == :block
        Base.remove_linenums!(expression)
        for assignment in expression.args
            if assignment.head == Symbol("const")
                @assert length(assignment.args) == 1
                assignment = assignment.args[1]
            end
            if assignment.head == :(=)
                parameter_name = assignment.args[1]
                parameter_value = assignment.args[2]
                push!(assignments,
                    :(const $(esc(parameter_name)) =
                    $(esc(parameter_value)))
                )
                push!(
                    dictionary_assignments,
                    :($(esc(dict))[$(string(parameter_name))] =
                        $(esc(parameter_name)))
                )
            else
                return :(throw(ErrorException(
                    "Only assignments are allowed inside @parameters block",
                )))
            end
        end
    elseif expression.head == Symbol("const")
        @assert length(expression.args) == 1
        expression = expression.args[1]
        parameter_name = expression.args[1]
        parameter_value = expression.args[2]
        push!(assignments,
            :(const $(esc(parameter_name)) =
            $(esc(parameter_value)))
        )
        push!(dictionary_assignments,
            :($(esc(dict))[$(string(parameter_name))] =
                $(esc(parameter_name)))
        )
    elseif expression.head == :(=)
        parameter_name = expression.args[1]
        parameter_value = expression.args[2]
        push!(assignments,
            :(const $(esc(parameter_name)) =
            $(esc(parameter_value)))
        )
        push!(dictionary_assignments,
            :($(esc(dict))[$(string(parameter_name))] =
                $(esc(parameter_name)))
        )
    else
        return :(throw(ErrorException("Expected either block or assignment!")))
    end
    assignments = Expr(:block, assignments...)
    dictionary_assignments = Expr(:block, dictionary_assignments...)
    return quote
        $(assignments)
        $(dictionary_assignments)
    end
end

macro experiment(params, log, expression, comment = :nothing)
    @assert expression.head == :(=) || expression.head == :block
    return quote
        local elapsed_time = Base.time_ns()
        local val = $(esc(expression))
        elapsed_time = Base.time_ns() - elapsed_time
        local params_experiment = copy($(esc(params)))
        local experiment = Experiment(val, params_experiment, $(esc(log)), $comment)
        elapsed_time = (
            string(round(Int, Float64(elapsed_time) / 1e9 ÷ 3600)) * "h "
          * string(round(Int, Float64(elapsed_time) / 1e9 % 3600 ÷ 60)) * "m "
          * string(Float64(elapsed_time) / 1e9 % 60) * "s")
        experiment.metadata["elapsed_time"] = elapsed_time
        save_experiment(experiment)
        val
    end
end
