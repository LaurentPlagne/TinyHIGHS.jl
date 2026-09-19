# MathOptInterface adapter for the revised simplex engine.
#
# The adapter deliberately keeps the MOI model representation separate from
# `SimplexEngine`: model construction follows the normal MOI protocol, while
# `optimize!` lowers the model to the compact CSC representation used by the
# native solver. This keeps the zero-allocation `SimplexEngine` path independent
# from MOI and gives JuMP a conventional, inspectable optimizer interface.

MOI.Utilities.@model(
    TinyHiGHSOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval),
    (),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (),
    (),
    true,
)

"""
    Optimizer()

MathOptInterface optimizer for continuous linear programs. The optimizer
supports scalar affine objectives and scalar affine constraints with equality,
one-sided, and interval bounds. Integer, quadratic, conic, and nonlinear
features are intentionally outside the scope of TinyHiGHS.

The model is built through the standard MOI API and lowered to a
`SimplexEngine` at `MOI.optimize!`. The low-level `SimplexEngine` API remains
the recommended entry point for allocation-free warm-start sequences.
"""
const MOIOptimizer = TinyHiGHSOptimizer{Float64}

const _MOI_ENGINE = :tinyhighs_engine
const _MOI_PRIMAL = :tinyhighs_primal
const _MOI_ROW_DUAL = :tinyhighs_row_dual
const _MOI_ROW_RECORDS = :tinyhighs_row_records
const _MOI_BOUND_RECORDS = :tinyhighs_bound_records
const _MOI_TERMINATION = :tinyhighs_termination
const _MOI_PRIMAL_STATUS = :tinyhighs_primal_status
const _MOI_DUAL_STATUS = :tinyhighs_dual_status
const _MOI_OBJECTIVE = :tinyhighs_objective
const _MOI_SOLVE_TIME = :tinyhighs_solve_time
const _MOI_RESULT_COUNT = :tinyhighs_result_count
const _MOI_REDUCED_COST = :tinyhighs_reduced_cost
const _MOI_SILENT = :tinyhighs_silent
const _MOI_TIME_LIMIT = :tinyhighs_time_limit
const _MOI_ITERATION_LIMIT = :tinyhighs_iteration_limit

function Optimizer()
    return _moi_reset_state!(MOIOptimizer())
end

function _moi_reset_state!(model::MOIOptimizer)
    empty!(model.ext)
    model.ext[_MOI_ENGINE] = nothing
    model.ext[_MOI_PRIMAL] = Float64[]
    model.ext[_MOI_ROW_DUAL] = Float64[]
    model.ext[_MOI_ROW_RECORDS] = Dict{MOI.ConstraintIndex,NamedTuple}()
    model.ext[_MOI_BOUND_RECORDS] = Dict{MOI.ConstraintIndex,NamedTuple}()
    model.ext[_MOI_TERMINATION] = MOI.OPTIMIZE_NOT_CALLED
    model.ext[_MOI_PRIMAL_STATUS] = MOI.NO_SOLUTION
    model.ext[_MOI_DUAL_STATUS] = MOI.NO_SOLUTION
    model.ext[_MOI_OBJECTIVE] = NaN
    model.ext[_MOI_SOLVE_TIME] = NaN
    model.ext[_MOI_RESULT_COUNT] = 0
    model.ext[_MOI_REDUCED_COST] = Float64[]
    model.ext[_MOI_SILENT] = false
    model.ext[_MOI_TIME_LIMIT] = nothing
    model.ext[_MOI_ITERATION_LIMIT] = kHighsIInf
    return model
end

function _moi_state!(model::MOIOptimizer)
    haskey(model.ext, _MOI_TERMINATION) || _moi_reset_state!(model)
    return model.ext
end

# GenericOptimizer implements the model-construction protocol. We only need to
# provide the optimizer-specific lifecycle and result attributes.
MOI.supports_incremental_interface(::MOIOptimizer) = true
MOI.supports(::MOIOptimizer, ::MOI.ObjectiveFunction{MOI.VariableIndex}) = true
MOI.supports(::MOIOptimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::MOIOptimizer, ::MOI.Silent) = true
MOI.supports(::MOIOptimizer, ::MOI.TimeLimitSec) = true
MOI.supports(::MOIOptimizer, ::MOI.VariablePrimal) = true
MOI.supports(::MOIOptimizer, ::MOI.ConstraintPrimal) = true
MOI.supports(::MOIOptimizer, ::MOI.ConstraintDual) = true
MOI.supports(::MOIOptimizer, ::MOI.ObjectiveValue) = true
MOI.supports(::MOIOptimizer, ::MOI.SimplexIterations) = true

function MOI.empty!(model::MOIOptimizer)
    MOI.empty!(model.objective)
    MOI.empty!(model.variables)
    MOI.empty!(model.constraints)
    empty!(model.var_to_name)
    model.name_to_var = nothing
    empty!(model.con_to_name)
    model.name_to_con = nothing
    _moi_reset_state!(model)
    return nothing
end

function MOI.set(model::MOIOptimizer, ::MOI.Silent, value::Bool)
    _moi_state!(model)[_MOI_SILENT] = value
    return nothing
end

MOI.get(model::MOIOptimizer, ::MOI.Silent) = get(_moi_state!(model), _MOI_SILENT, false)

function MOI.set(model::MOIOptimizer, ::MOI.TimeLimitSec, value::Union{Nothing,Real})
    value === nothing || value >= 0 || throw(ArgumentError("TimeLimitSec must be non-negative"))
    _moi_state!(model)[_MOI_TIME_LIMIT] = value === nothing ? nothing : Float64(value)
    return nothing
end

MOI.get(model::MOIOptimizer, ::MOI.TimeLimitSec) = get(_moi_state!(model), _MOI_TIME_LIMIT, nothing)

function MOI.set(model::MOIOptimizer, attr::MOI.RawOptimizerAttribute, value)
    state = _moi_state!(model)
    if attr.name == "simplex_iteration_limit"
        value isa Integer || throw(ArgumentError("simplex_iteration_limit must be an integer"))
        state[_MOI_ITERATION_LIMIT] = Int(value)
    else
        state[attr.name] = value
    end
    return nothing
end

function MOI.get(model::MOIOptimizer, attr::MOI.RawOptimizerAttribute)
    state = _moi_state!(model)
    if attr.name == "simplex_iteration_limit"
        return get(state, _MOI_ITERATION_LIMIT, kHighsIInf)
    elseif attr.name == "simplex_iteration_count"
        engine = get(state, _MOI_ENGINE, nothing)
        return engine === nothing ? 0 : engine.iteration_count
    elseif attr.name == "simplex_engine"
        return get(state, _MOI_ENGINE, nothing)
    end
    return get(state, attr.name, nothing)
end

function _moi_set_bounds!(lower::Vector{Float64}, upper::Vector{Float64},
    model::MOIOptimizer, varmap::Dict{MOI.VariableIndex,Int})
    records = Dict{MOI.ConstraintIndex,NamedTuple}()
    for (S, kind) in ((MOI.GreaterThan{Float64}, :lower),
                      (MOI.LessThan{Float64}, :upper),
                      (MOI.EqualTo{Float64}, :equal),
                      (MOI.Interval{Float64}, :interval))
        for ci in MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
            vi = MOI.get(model, MOI.ConstraintFunction(), ci)
            j = varmap[vi]
            set = MOI.get(model, MOI.ConstraintSet(), ci)
            if kind === :lower
                lower[j] = set.lower
            elseif kind === :upper
                upper[j] = set.upper
            elseif kind === :equal
                lower[j] = set.value
                upper[j] = set.value
            else
                lower[j] = set.lower
                upper[j] = set.upper
            end
            records[ci] = (; variable=j, set, kind)
        end
    end
    return records
end

function _moi_objective(model::MOIOptimizer, varmap::Dict{MOI.VariableIndex,Int}, n::Int)
    costs = zeros(Float64, n)
    offset = 0.0
    objective_type = MOI.get(model, MOI.ObjectiveFunctionType())
    if objective_type == MOI.VariableIndex
        vi = MOI.get(model, MOI.ObjectiveFunction{MOI.VariableIndex}())
        costs[varmap[vi]] = 1.0
    elseif objective_type == MOI.ScalarAffineFunction{Float64}
        f = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        offset = f.constant
        for term in f.terms
            costs[varmap[term.variable]] += term.coefficient
        end
    elseif objective_type != MOI.ScalarAffineFunction{Float64}
        throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunctionType()))
    end
    return costs, offset
end

function _moi_model_to_lp(model::MOIOptimizer)
    variables = MOI.get(model, MOI.ListOfVariableIndices())
    varmap = Dict{MOI.VariableIndex,Int}(vi => i for (i, vi) in enumerate(variables))
    n = length(variables)
    lower = fill(-kHighsInf, n)
    upper = fill(kHighsInf, n)
    bound_records = _moi_set_bounds!(lower, upper, model, varmap)

    row_records = Dict{MOI.ConstraintIndex,NamedTuple}()
    rows = NamedTuple[]
    columns = [Dict{Int,Float64}() for _ in 1:n]
    row = 0
    for (S, kind) in ((MOI.GreaterThan{Float64}, :lower),
                      (MOI.LessThan{Float64}, :upper),
                      (MOI.EqualTo{Float64}, :equal),
                      (MOI.Interval{Float64}, :interval))
        for ci in MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64},S}())
            row += 1
            f = MOI.get(model, MOI.ConstraintFunction(), ci)
            set = MOI.get(model, MOI.ConstraintSet(), ci)
            rl, ru = if kind === :lower
                set.lower - f.constant, kHighsInf
            elseif kind === :upper
                -kHighsInf, set.upper - f.constant
            elseif kind === :equal
                set.value - f.constant, set.value - f.constant
            else
                set.lower - f.constant, set.upper - f.constant
            end
            for term in f.terms
                j = varmap[term.variable]
                columns[j][row] = get(columns[j], row, 0.0) + term.coefficient
            end
            record = (; row, func=f, set, kind)
            row_records[ci] = record
            push!(rows, record)
        end
    end

    starts = Vector{Int}(undef, n + 1)
    indices = Int[]
    values = Float64[]
    starts[1] = 1
    for j in 1:n
        for i in sort!(collect(keys(columns[j])))
            value = columns[j][i]
            iszero(value) && continue
            push!(indices, i)
            push!(values, value)
        end
        starts[j + 1] = length(indices) + 1
    end
    matrix = SparseMatrix(n, row, starts, indices, values)
    costs, offset = _moi_objective(model, varmap, n)
    sense = MOI.get(model, MOI.ObjectiveSense()) == MOI.MAX_SENSE ? kMaximize : kMinimize
    row_lower = Float64[]
    row_upper = Float64[]
    for record in rows
        if record.kind === :lower
            push!(row_lower, record.set.lower - record.func.constant)
            push!(row_upper, kHighsInf)
        elseif record.kind === :upper
            push!(row_lower, -kHighsInf)
            push!(row_upper, record.set.upper - record.func.constant)
        elseif record.kind === :equal
            bound = record.set.value - record.func.constant
            push!(row_lower, bound)
            push!(row_upper, bound)
        else
            push!(row_lower, record.set.lower - record.func.constant)
            push!(row_upper, record.set.upper - record.func.constant)
        end
    end
    lp = SimplexLp(n, row, matrix, costs, lower, upper,
        row_lower, row_upper; offset=offset, sense=sense)
    return lp, row_records, bound_records
end

function _moi_primal(engine::SimplexEngine)
    n = engine.lp.num_col
    x = copy(engine.info.workValue[1:n])
    for (row, variable) in enumerate(engine.basis.basicIndex)
        variable <= n || continue
        x[variable] = engine.info.baseValue[row]
    end
    return x
end

function _moi_solve_without_rows!(state, lp::SimplexLp)
    n = lp.num_col
    x = zeros(Float64, n)
    reduced = [Int(lp.sense) * lp.col_cost[j] for j in 1:n]
    any(lp.col_lower[j] > lp.col_upper[j] for j in 1:n) &&
        return kInfeasible, x, reduced, NaN
    unbounded = false
    for j in 1:n
        c = lp.col_cost[j]
        if lp.sense === kMinimize
            if c > 0.0
                if !isfinite(lp.col_lower[j])
                    unbounded = true
                else
                    x[j] = lp.col_lower[j]
                end
            elseif c < 0.0
                if !isfinite(lp.col_upper[j])
                    unbounded = true
                else
                    x[j] = lp.col_upper[j]
                end
            elseif isfinite(lp.col_lower[j])
                x[j] = lp.col_lower[j]
            elseif isfinite(lp.col_upper[j])
                x[j] = lp.col_upper[j]
            end
        else
            if c > 0.0
                if !isfinite(lp.col_upper[j])
                    unbounded = true
                else
                    x[j] = lp.col_upper[j]
                end
            elseif c < 0.0
                if !isfinite(lp.col_lower[j])
                    unbounded = true
                else
                    x[j] = lp.col_lower[j]
                end
            elseif isfinite(lp.col_lower[j])
                x[j] = lp.col_lower[j]
            elseif isfinite(lp.col_upper[j])
                x[j] = lp.col_upper[j]
            end
        end
    end
    if unbounded
        return kUnbounded, x, reduced, NaN
    end
    objective = lp.offset + sum(lp.col_cost[j] * x[j] for j in 1:n; init=0.0)
    return kOptimal, x, reduced, objective
end

function _moi_termination(status::ModelStatus)
    status === kOptimal && return MOI.OPTIMAL
    status === kInfeasible && return MOI.INFEASIBLE
    status === kUnbounded && return MOI.DUAL_INFEASIBLE
    status === kUnboundedOrInfeasible && return MOI.INFEASIBLE_OR_UNBOUNDED
    status === kTimeLimit && return MOI.TIME_LIMIT
    status === kIterationLimit && return MOI.ITERATION_LIMIT
    return MOI.OTHER_ERROR
end

function MOI.optimize!(model::MOIOptimizer)
    state = _moi_state!(model)
    t0 = time_ns()
    lp, row_records, bound_records = _moi_model_to_lp(model)
    options = SimplexOptions(; time_limit=get(state, _MOI_TIME_LIMIT, nothing) === nothing ?
        kHighsInf : get(state, _MOI_TIME_LIMIT),
        simplex_iteration_limit=get(state, _MOI_ITERATION_LIMIT, kHighsIInf))
    engine = nothing
    if lp.num_row == 0
        status, primal, reduced, objective = _moi_solve_without_rows!(state, lp)
    else
        engine = SimplexEngine(lp, options)
        status = solve!(engine)
        primal = status === kOptimal ? _moi_primal(engine) : Float64[]
        reduced = status === kOptimal ? copy(engine.info.workDual[1:lp.num_col]) : Float64[]
        objective = status === kOptimal ? engine.info.primal_objective_value : NaN
    end
    termination = _moi_termination(status)
    state[_MOI_ENGINE] = engine
    state[_MOI_PRIMAL] = status === kOptimal ? primal : Float64[]
    state[_MOI_ROW_RECORDS] = row_records
    state[_MOI_BOUND_RECORDS] = bound_records
    state[_MOI_ROW_DUAL] = status === kOptimal && engine !== nothing ?
        [Int(lp.sense) * (-engine.info.workDual[lp.num_col + i]) for i in 1:lp.num_row] : Float64[]
    state[_MOI_REDUCED_COST] = reduced
    state[_MOI_TERMINATION] = termination
    state[_MOI_PRIMAL_STATUS] = status === kOptimal ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
    state[_MOI_DUAL_STATUS] = status === kOptimal ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
    state[_MOI_OBJECTIVE] = status === kOptimal ? objective : NaN
    state[_MOI_SOLVE_TIME] = (time_ns() - t0) / 1.0e9
    state[_MOI_RESULT_COUNT] = status === kOptimal ? 1 : 0
    return nothing
end

function MOI.get(model::MOIOptimizer, ::MOI.SolverName)
    return "TinyHiGHS"
end

MOI.get(model::MOIOptimizer, ::MOI.SolverVersion) = string(VERSION)
MOI.get(model::MOIOptimizer, ::MOI.TerminationStatus) = _moi_state!(model)[_MOI_TERMINATION]
MOI.get(model::MOIOptimizer, ::MOI.PrimalStatus) = _moi_state!(model)[_MOI_PRIMAL_STATUS]
MOI.get(model::MOIOptimizer, ::MOI.DualStatus) = _moi_state!(model)[_MOI_DUAL_STATUS]
MOI.get(model::MOIOptimizer, ::MOI.ResultCount) = _moi_state!(model)[_MOI_RESULT_COUNT]
MOI.get(model::MOIOptimizer, ::MOI.SolveTimeSec) = _moi_state!(model)[_MOI_SOLVE_TIME]
MOI.get(model::MOIOptimizer, ::MOI.SimplexIterations) = begin
    engine = _moi_state!(model)[_MOI_ENGINE]
    engine === nothing ? 0 : engine.iteration_count
end

function MOI.get(model::MOIOptimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return _moi_state!(model)[_MOI_OBJECTIVE]
end

function MOI.get(model::MOIOptimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    primal = _moi_state!(model)[_MOI_PRIMAL]
    return primal[vi.value]
end

function _moi_constraint_value(model::MOIOptimizer, ci, x::Vector{Float64})
    f = MOI.get(model, MOI.ConstraintFunction(), ci)
    f isa MOI.VariableIndex && return x[f.value]
    return f.constant + sum(term.coefficient * x[term.variable.value] for term in f.terms; init=0.0)
end

function MOI.get(model::MOIOptimizer, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(model, attr)
    return _moi_constraint_value(model, ci, _moi_state!(model)[_MOI_PRIMAL])
end

function _moi_bound_dual(model::MOIOptimizer, record)
    state = _moi_state!(model)
    reduced = state[_MOI_REDUCED_COST][record.variable]
    set = record.set
    if set isa MOI.GreaterThan
        return max(reduced, 0.0)
    elseif set isa MOI.LessThan
        return min(reduced, 0.0)
    elseif set isa MOI.Interval
        return reduced >= 0.0 ? reduced : 0.0
    end
    return reduced
end

function MOI.get(model::MOIOptimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(model, attr)
    state = _moi_state!(model)
    if haskey(state[_MOI_ROW_RECORDS], ci)
        return state[_MOI_ROW_DUAL][state[_MOI_ROW_RECORDS][ci].row]
    end
    if haskey(state[_MOI_BOUND_RECORDS], ci)
        return _moi_bound_dual(model, state[_MOI_BOUND_RECORDS][ci])
    end
    throw(MOI.InvalidIndex(ci))
end
