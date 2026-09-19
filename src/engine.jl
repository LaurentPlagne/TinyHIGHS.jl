# Port of `simplex/HEkk.{h,cpp}` (MIT License, HiGHS).
#
# Core revised simplex state, values, factorization, updates, and solve routines:
# `setBasis` (logical basis), `setNonbasicMove`, `initialiseLpColBound`/
# `initialiseLpRowBound`/`initialiseBound`, `initialiseLpColCost`/
# `initialiseLpRowCost`/`initialiseCost`, `initialiseNonbasicValueAndMove`,
# `computePrimal`, `computeDual`, `fullBtran`/`fullPrice`, primal and dual objectives,
# basis rebuilding and reinversion, dual steepest edge (DSE) weights.
#
# The solved system follows the HiGHS convention: each row carries a slack
# (logical) variable `s = -a_i x`, with bounds `[-row_upper, -row_lower]`, such that
# `[A I] [x; s] = 0` — the RHS of the LP is represented via the bounds of the
# logicals, not via a separate vector b.

"""
    SimplexLp(num_col, num_row, a_matrix, col_cost, col_lower, col_upper,
              row_lower, row_upper; offset = 0.0, sense = kMinimize)

Linear programming model definition.

# Fields
- `num_col::Int`: Number of structural columns (variables).
- `num_row::Int`: Number of constraints (rows).
- `a_matrix::SparseMatrix`: Constraint matrix in 1-based column-wise (CSC) representation.
- `col_cost::Vector{Float64}`: Objective linear coefficients (length `num_col`).
- `col_lower::Vector{Float64}`: Variable lower bounds (length `num_col`).
- `col_upper::Vector{Float64}`: Variable upper bounds (length `num_col`).
- `row_lower::Vector{Float64}`: Constraint lower bounds (length `num_row`).
- `row_upper::Vector{Float64}`: Constraint upper bounds (length `num_row`).
- `offset::Float64`: Objective constant offset (defaults to `0.0`).
- `sense::ObjSense`: Optimization direction (`kMinimize` or `kMaximize`).
- `scale::Scale`: Optional scaling factors.
- `is_scaled::Bool`: Whether the model is currently scaled.
"""
mutable struct SimplexLp
    num_col::Int
    num_row::Int
    a_matrix::SparseMatrix
    col_cost::Vector{Float64}
    col_lower::Vector{Float64}
    col_upper::Vector{Float64}
    row_lower::Vector{Float64}
    row_upper::Vector{Float64}
    offset::Float64
    sense::ObjSense
    scale::Scale
    is_scaled::Bool
end

function SimplexLp(num_col::Int, num_row::Int, a_matrix::SparseMatrix,
    col_cost::Vector{Float64}, col_lower::Vector{Float64},
    col_upper::Vector{Float64}, row_lower::Vector{Float64},
    row_upper::Vector{Float64}; offset::Float64=0.0,
    sense::ObjSense=kMinimize)
    (num_col >= 0 && num_row >= 0) || throw(ArgumentError("dimensions must be non-negative"))
    (a_matrix.num_col == num_col && a_matrix.num_row == num_row) ||
        throw(ArgumentError("dimensions of a_matrix mismatch num_col/num_row"))
    is_colwise(a_matrix) ||
        throw(ArgumentError("a_matrix must be column-wise"))
    (length(col_cost) == num_col && length(col_lower) == num_col &&
     length(col_upper) == num_col) ||
        throw(ArgumentError("column vector lengths mismatch num_col"))
    (length(row_lower) == num_row && length(row_upper) == num_row) ||
        throw(ArgumentError("row vector lengths mismatch num_row"))
    return SimplexLp(num_col, num_row, a_matrix, col_cost, col_lower, col_upper,
        row_lower, row_upper, offset, sense, Scale(), false)
end

"""
    SimplexOptions(; ...)

Configuration parameters for the simplex engine, matching HiGHS default settings.

Key options:
- `primal_feasibility_tolerance`: Primal tolerance (default `1e-7`).
- `dual_feasibility_tolerance`: Dual tolerance (default `1e-7`).
- `time_limit`: Time limit in seconds (default `Inf`).
- `simplex_iteration_limit`: Maximum number of simplex iterations.
- `simplex_dual_edge_weight_strategy`: Dual pricing edge weight strategy (`kSimplexEdgeWeightStrategyChoose`, `kSimplexEdgeWeightStrategyDantzig`, `kSimplexEdgeWeightStrategyDevex`, `kSimplexEdgeWeightStrategySteepestEdge`).
- `simplex_update_limit`: Maximum number of basis updates before full refactorization (default `5000`).
- `random_seed`: Seed for deterministic pseudo-random number generator (perturbations).
"""
struct SimplexOptions
    cost_scale_factor::Int
    dual_simplex_cost_perturbation_multiplier::Float64
    primal_simplex_bound_perturbation_multiplier::Float64
    primal_feasibility_tolerance::Float64
    dual_feasibility_tolerance::Float64
    small_matrix_value::Float64
    time_limit::Float64
    simplex_iteration_limit::Int
    objective_bound::Float64
    objective_target::Float64
    simplex_dual_edge_weight_strategy::Int
    simplex_primal_edge_weight_strategy::Int
    simplex_scale_strategy::Int
    allowed_matrix_scale_factor::Int
    simplex_price_strategy::Int
    simplex_update_limit::Int
    max_dual_simplex_cleanup_level::Int
    max_dual_simplex_phase1_cleanup_level::Int
    no_unnecessary_rebuild_refactor::Bool
    rebuild_refactor_solution_error_tolerance::Float64
    dual_simplex_pivot_growth_tolerance::Float64
    dual_steepest_edge_weight_log_error_threshold::Float64
    random_seed::Int
end

SimplexOptions(; cost_scale_factor::Int=0,
    dual_simplex_cost_perturbation_multiplier::Float64=1.0,
    primal_simplex_bound_perturbation_multiplier::Float64=1.0,
    primal_feasibility_tolerance::Float64=1e-7,
    dual_feasibility_tolerance::Float64=1e-7,
    small_matrix_value::Float64=1e-9,
    time_limit::Float64=kHighsInf,
    simplex_iteration_limit::Int=kHighsIInf,
    objective_bound::Float64=kHighsInf,
    objective_target::Float64=kHighsInf,
    simplex_dual_edge_weight_strategy::Int=kSimplexEdgeWeightStrategyChoose,
    simplex_primal_edge_weight_strategy::Int=kSimplexEdgeWeightStrategyChoose,
    simplex_scale_strategy::Int=kSimplexScaleStrategyOff,
    allowed_matrix_scale_factor::Int=kDefaultAllowedMatrixPow2Scale,
    simplex_price_strategy::Int=kSimplexPriceStrategyRowSwitchColSwitch,
    simplex_update_limit::Int=5000,
    max_dual_simplex_cleanup_level::Int=1,
    max_dual_simplex_phase1_cleanup_level::Int=2,
    no_unnecessary_rebuild_refactor::Bool=true,
    rebuild_refactor_solution_error_tolerance::Float64=1e-8,
    dual_simplex_pivot_growth_tolerance::Float64=1e-9,
    dual_steepest_edge_weight_log_error_threshold::Float64=10.0,
    random_seed::Int=0) =
    SimplexOptions(cost_scale_factor, dual_simplex_cost_perturbation_multiplier,
        primal_simplex_bound_perturbation_multiplier,
        primal_feasibility_tolerance, dual_feasibility_tolerance,
        small_matrix_value, time_limit, simplex_iteration_limit,
        objective_bound, objective_target, simplex_dual_edge_weight_strategy,
        simplex_primal_edge_weight_strategy, simplex_scale_strategy,
        allowed_matrix_scale_factor, simplex_price_strategy,
        simplex_update_limit,
        max_dual_simplex_cleanup_level, max_dual_simplex_phase1_cleanup_level,
        no_unnecessary_rebuild_refactor,
        rebuild_refactor_solution_error_tolerance,
        dual_simplex_pivot_growth_tolerance,
        dual_steepest_edge_weight_log_error_threshold, random_seed)

"""
    SimplexEngine(lp, options = SimplexOptions(); basis = nothing)

Core revised simplex solver engine equivalent to HiGHS's `HEkk`.

Encapsulates:
- The linear model (`lp::SimplexLp`).
- Solver configuration options (`options::SimplexOptions`).
- Current basis state (`basis::SimplexBasis`).
- Sized persistent working buffers (`info::SimplexInfo`), reused across solves without reallocating.
- Numerical Linear Algebra and basis LU factorization state (`nla::Nla`).
- Pseudo-random number generator for numerical perturbation (`random::HighsRandom`).

# Warm-Starting
`SimplexEngine` is designed for high-frequency repeated solves: modify bounds or costs using
`change_col_bounds!`, `change_row_bounds!`, `change_cols_cost!` and call `solve!(engine)` to
perform warm-start re-optimization with zero memory allocation.
"""
mutable struct SimplexEngine
    lp::SimplexLp
    options::SimplexOptions
    basis::SimplexBasis
    info::SimplexInfo
    nla::Nla
    cost_scale::Float64
    status::SimplexStatus
    random::HighsRandom
    iteration_count::Int
    # Current solve iteration counter: `HApp::solveLpSimplex` copies
    # `highs_info.simplex_iteration_count` (reset to zero by `Highs::run`)
    # into `ekk_instance.iteration_count_` on each resolve, making
    # `simplex_iteration_limit` apply **per solve** while `iteration_count`
    # in this port is cumulative (like `highs_info`, used for reporting).
    iteration_count0::Int
    total_synthetic_tick::Float64
    build_synthetic_tick::Float64
    dual_edge_weight::Vector{Float64}
    scattered_dual_edge_weight::Vector{Float64}
    ar_matrix::SparseMatrix
    model_status::ModelStatus
    solve_bailout::Bool
    solve_start_time::Float64
    dual_simplex_cleanup_level::Int
    dual_simplex_phase1_cleanup_level::Int
    visited_basis::Set{UInt64}
    previous_iteration_cycling_detected::Int
    bad_basis_change::Vector{BadBasisChange}
    primal_ray_record::RayRecord
    # `Highs::run`: significantly inconsistent bounds lead to declaring the LP
    # infeasible without invoking the simplex (flag set by `initialise_for_solve!`).
    bounds_infeasible::Bool
    # `HighsBasis basis_` from Highs level: basis returned by the last solve,
    # refreshed at the end of each resolve. Used for warm starting when the Ekk
    # basis was invalidated without the model changing its basis (coefficient
    # modification: `HEkk::clear` followed by `HEkk::setBasis(basis_)`).
    highs_basis::SimplexBasis
    highs_basis_valid::Bool
    primal_col::HVector
    dual_col::HVector
    dual_row::HVector
    basic_index_before::Vector{Int}
    fse_solution_value::Vector{Float64}
    fse_solution_index::Vector{Int}
    fse_solution_nonzero::Vector{Bool}
    fse_btran_scattered::Vector{Float64}
    fse_random::HighsRandom
end

function SimplexEngine(lp::SimplexLp, options::SimplexOptions=SimplexOptions();
    basis::Union{Nothing,SimplexBasis}=nothing)
    provided_basis = basis !== nothing
    if !provided_basis
        basis = SimplexBasis(lp.num_col, lp.num_row)
    else
        (length(basis.basicIndex) == lp.num_row &&
         length(basis.nonbasicFlag) == lp.num_col + lp.num_row &&
         length(basis.nonbasicMove) == lp.num_col + lp.num_row) ||
            throw(ArgumentError("basis dimensions incompatible with LP dimensions"))
    end
    info = SimplexInfo(lp.num_col, lp.num_row)
    factor = HFactor(lp.num_col, lp.num_row, lp.num_row, lp.a_matrix.start,
        lp.a_matrix.index, lp.a_matrix.value, basis.basicIndex)
    basis.basicIndex = factor.basic_index
    status = SimplexStatus()
    # A provided basis is valid: `initialise_for_solve!` must not replace it
    # with the logical basis. Its hash is derived from `nonbasicFlag`.
    status.has_basis = provided_basis
    provided_basis && (basis.hash = basis_hash(basis))
    engine = SimplexEngine(lp, options, basis, info,
        Nla(factor, lp.num_col, lp.num_row), 1.0, status,
        HighsRandom(options.random_seed), 0, 0, 0.0, 0.0,
        ones(lp.num_row), zeros(lp.num_col + lp.num_row),
        SparseMatrix(lp.num_col, lp.num_row), kNotset, false, 0.0, 0, 0,
        Set{UInt64}(), -kHighsIInf, BadBasisChange[], RayRecord(), false,
        SimplexBasis(lp.num_col, lp.num_row), false,
        HVector(lp.num_row), HVector(lp.num_row), HVector(lp.num_col),
        zeros(Int, lp.num_row), zeros(50), zeros(Int, 50),
        zeros(Bool, lp.num_row), zeros(lp.num_col + lp.num_row),
        HighsRandom(1))
    # `initialiseEkk`: internal options, RNG reinitialized, then initial draw
    # of random vectors (`initialiseSimplexLpRandomVectors`).
    set_simplex_options!(engine)
    initialise_simplex_lp_random_vectors!(engine)
    return engine
end

"""
`HEkk::setSimplexOptions` and `updateSimplexOptions` — static copy of options
into workspace (the port does not modify options during solve).
"""
function set_simplex_options!(e::SimplexEngine)
    info = e.info
    info.simplex_strategy = kSimplexStrategyDualPlain
    info.primal_simplex_bound_perturbation_multiplier =
        e.options.primal_simplex_bound_perturbation_multiplier
    info.dual_simplex_cost_perturbation_multiplier =
        e.options.dual_simplex_cost_perturbation_multiplier
    info.dual_edge_weight_strategy = e.options.simplex_dual_edge_weight_strategy
    info.price_strategy = e.options.simplex_price_strategy
    # Perturbation multipliers are copied into `info` as in the C++ source:
    # crossover cleanup paths (dual → primal, primal → dual) set them to zero
    # for the duration of a solve.
    info.factor_pivot_threshold = kDefaultPivotThreshold
    info.update_limit = e.options.simplex_update_limit
    return e
end

"""
    repaired_bounds(lower, upper, tolerance)

Lambda `infeasibleBoundOk` from `Highs::infeasibleBoundsOk`: for inconsistent
bounds (`lower > upper`) whose gap is within primal feasibility tolerance,
the integer bound (in the sense `x == round(x)`) is preserved and the other
follows, otherwise both are set to their midpoint. Returns `(ok, lower, upper)`.
"""
function repaired_bounds(lower::Float64, upper::Float64, tolerance::Float64)
    (upper - lower) > -tolerance || return false, lower, upper
    integer_lower = lower == floor(lower + 0.5)
    integer_upper = upper == floor(upper + 0.5)
    if integer_lower
        return true, lower, lower
    elseif integer_upper
        return true, upper, upper
    end
    mid = 0.5 * (lower + upper)
    return true, mid, mid
end

"""
    infeasible_bounds_ok!(e)

`Highs::infeasibleBoundsOk` (without integrality or reporting): repairs in-place
inconsistent bounds within tolerance and counts significant inconsistencies.
Returns `false` if any remain: `Highs::run` then declares `kInfeasible` without
solving. Row bounds are checked identically to column bounds.
"""
function infeasible_bounds_ok!(e::SimplexEngine)
    lp = e.lp
    tolerance = e.options.primal_feasibility_tolerance
    num_true_infeasible_bound = 0
    for iCol ∈ 1:lp.num_col
        lower = lp.col_lower[iCol]
        upper = lp.col_upper[iCol]
        lower > upper || continue
        ok, lower, upper = repaired_bounds(lower, upper, tolerance)
        if ok
            lp.col_lower[iCol] = lower
            lp.col_upper[iCol] = upper
        else
            num_true_infeasible_bound += 1
        end
    end
    for iRow ∈ 1:lp.num_row
        lower = lp.row_lower[iRow]
        upper = lp.row_upper[iRow]
        lower > upper || continue
        ok, lower, upper = repaired_bounds(lower, upper, tolerance)
        if ok
            lp.row_lower[iRow] = lower
            lp.row_upper[iRow] = upper
        else
            num_true_infeasible_bound += 1
        end
    end
    return num_true_infeasible_bound == 0
end

"""
    initialise_for_solve!(e)

`HEkk::initialiseForSolve`: internal options, pseudo-random vectors (consumed
identically to C++ source), logical basis and factor if absent, row-wise matrix,
costs/bounds/values, primal, dual, infeasibilities, and objectives.
Status is set to `kOptimal` if the starting point is already primal and dual
feasible, and to `kInfeasible` (without simplex initialization) if `Highs::run`
rejects bounds.
"""
function initialise_for_solve!(e::SimplexEngine)
    # `HApp::solveLpSimplex`: solve iteration work counter restarts from
    # `highs_info.simplex_iteration_count` (zero at the start of each `Highs_run`)
    # — this makes `simplex_iteration_limit` apply **per solve**.
    e.iteration_count0 = e.iteration_count
    # `Highs::run`: inconsistent bounds are repaired or render the LP
    # infeasible BEFORE simplex is invoked (neither options nor RNG touched).
    if !infeasible_bounds_ok!(e)
        e.bounds_infeasible = true
        e.model_status = kInfeasible
        return e
    end
    e.bounds_infeasible = false
    # `HApp::solveLpSimplex`: LP scaling before `moveLp` (the LP may be
    # rescaled, known scale factors reapplied, or cleared).
    consider_scaling!(e)
    # `HEkk::setNlaPointersForLpAndScale`: NLA conversions are only needed
    # when bridging an unscaled LP with a scaled factor. Here the LP is scaled
    # and so is the factor: scales are already in the model (and in
    # `apply_lp_scale!`), conversions must be inactive (`scale_ == NULL`).
    e.nla.scale = e.lp.scale.has_scaling && !e.lp.is_scaled ? e.lp.scale :
                  nothing
    set_simplex_options!(e)
    # `HEkk::solve`: `initialiseControl` before `initialiseForSolve` (dual
    # also calls it in `solve!`, without effect in between).
    initialise_control!(e)
    initialise_simplex_lp_random_vectors!(e)
    if !e.status.has_basis
        if e.highs_basis_valid
            # `HApp::solveLpSimplex`: Ekk basis cleared but `basis_` valid
            # (coefficient change) — `HEkk::setBasis`.
            restore_basis!(e, e.highs_basis)
        else
            set_basis!(e)
        end
    end
    if !e.status.has_invert
        rank_deficiency = compute_factor!(e)
        if rank_deficiency != 0
            # Singular starting basis: factor completed with logicals;
            # synchronize basis and proceed (like `initialiseSimplexLpBasisAndFactor`).
            handle_rank_deficiency!(e)
            set_nonbasic_move!(e)
            e.status.has_basis = true
            e.status.has_invert = true
            e.status.has_fresh_invert = true
        end
    end
    initialise_partitioned_rowwise_matrix!(e)
    initialise_cost!(e, kPrimal, kSolvePhaseUnknown)
    initialise_bound!(e, kPrimal, kSolvePhaseUnknown)
    initialise_nonbasic_value_and_move!(e)
    compute_primal!(e)
    compute_dual!(e)
    compute_simplex_infeasible!(e)
    compute_dual_objective_value!(e)
    compute_primal_objective_value!(e)
    # As in `HEkk::initialiseForSolve`: status reset to `kNotset` before optimality
    # test (otherwise a `kOptimal` from a previous solve survives on a modified
    # model and the solve is skipped).
    e.model_status = kNotset
    if e.info.num_primal_infeasibilities == 0 &&
       e.info.num_dual_infeasibilities == 0
        e.model_status = kOptimal
    end
    empty!(e.visited_basis)
    push!(e.visited_basis, e.basis.hash)
    e.previous_iteration_cycling_detected = -kHighsIInf
    return e
end

##############################################################################
# LP modifications between resolves
#
# Source contract: modifications pass through the LP (`HighsLp`), then
# `HEkk::updateStatus(LpAction)` invalidates necessary state. A bound or cost
# change **preserves the basis** (warm start); a matrix coefficient change
# wipes all state (`HEkk::clear`, next solve restarts from logical basis).
##############################################################################

"""`HEkk::invalidateBasisArtifacts` — clears basis, factor, and associated caches."""
function invalidate_basis_artifacts!(e::SimplexEngine)
    e.status.has_ar_matrix = false
    e.status.has_dual_steepest_edge_weights = false
    e.status.has_invert = false
    e.status.has_fresh_invert = false
    e.status.has_fresh_rebuild = false
    e.status.has_dual_objective_value = false
    e.status.has_primal_objective_value = false
    clear_ray_records!(e)
    return e
end

"""`HEkk::invalidateBasis`."""
function invalidate_basis!(e::SimplexEngine)
    e.status.has_basis = false
    invalidate_basis_artifacts!(e)
    return e
end

"""`HEkk::invalidateBasisMatrix`."""
function invalidate_basis_matrix!(e::SimplexEngine)
    invalidate_basis!(e)
    return e
end

"""
`HEkk::clear` (without the LP, which remains the mutated model): simplex state
is cleared, next `initialise_for_solve!` restarts from the logical basis.
"""
function clear_ekk!(e::SimplexEngine)
    invalidate_basis_matrix!(e)
    empty!(e.bad_basis_change)
    empty!(e.visited_basis)
    e.previous_iteration_cycling_detected = -kHighsIInf
    return e
end

"""
    update_status!(e, action)

`HEkk::updateStatus`: handles side effects of LP modifications.

- `kLpActionScale`: basis and NLA become invalid (scaled space changed);
- `kLpActionNewCosts`, `kLpActionNewBounds`: basis is preserved (warm start),
  but rebuild and objective values must be recomputed;
- `kLpActionNewBasis`: basis is discarded;
- `kLpActionNewRows`: entire state is discarded (matrix coefficient change).
"""
function update_status!(e::SimplexEngine, action::Int)
    if action == kLpActionScale
        invalidate_basis_matrix!(e)
    elseif action == kLpActionNewCosts || action == kLpActionNewBounds
        e.status.has_fresh_rebuild = false
        e.status.has_dual_objective_value = false
        e.status.has_primal_objective_value = false
    elseif action == kLpActionNewBasis
        invalidate_basis!(e)
    elseif action == kLpActionNewRows
        clear_ekk!(e)
    else
        error("SimplexEngine: LP action $action not implemented")
    end
    return e
end

# Inconsistent bounds are not rejected here: `assessBounds` in C++ source
# simply issues a warning and lets the LP carry them; `infeasible_bounds_ok!`
# repairs them or declares the LP infeasible during solve.

"""
    change_col_bounds!(e::SimplexEngine, iCol::Int, lower::Float64, upper::Float64)

Update lower and upper bounds of a single column variable `iCol` (1-based) in-place.
Signals the engine that bounds have changed while keeping the basis intact for warm-starting.
"""
function change_col_bounds!(e::SimplexEngine, iCol::Int, lower::Float64,
    upper::Float64)
    (1 <= iCol <= e.lp.num_col) || throw(ArgumentError("column $iCol out of bounds"))
    e.lp.col_lower[iCol] = lower
    e.lp.col_upper[iCol] = upper
    update_status!(e, kLpActionNewBounds)
    return e
end

"""
    change_cols_bounds!(e::SimplexEngine, iCols, lowers, uppers)

Update bounds for a collection of columns in-place.
"""
function change_cols_bounds!(e::SimplexEngine, iCols::AbstractVector{Int},
    lowers::AbstractVector{Float64}, uppers::AbstractVector{Float64})
    (length(iCols) == length(lowers) == length(uppers)) ||
        throw(ArgumentError("dimension mismatch in change_cols_bounds!"))
    for k ∈ eachindex(iCols)
        iCol = iCols[k]
        (1 <= iCol <= e.lp.num_col) ||
            throw(ArgumentError("column $iCol out of bounds"))
        e.lp.col_lower[iCol] = lowers[k]
        e.lp.col_upper[iCol] = uppers[k]
    end
    update_status!(e, kLpActionNewBounds)
    return e
end

"""
    change_row_bounds!(e::SimplexEngine, iRow::Int, lower::Float64, upper::Float64)

Update lower and upper bounds of a single constraint row `iRow` (1-based) in-place.
"""
function change_row_bounds!(e::SimplexEngine, iRow::Int, lower::Float64,
    upper::Float64)
    (1 <= iRow <= e.lp.num_row) || throw(ArgumentError("row $iRow out of bounds"))
    e.lp.row_lower[iRow] = lower
    e.lp.row_upper[iRow] = upper
    update_status!(e, kLpActionNewBounds)
    return e
end

"""
    change_rows_bounds!(e::SimplexEngine, iRows, lowers, uppers)

Update bounds for a collection of constraint rows in-place.
"""
function change_rows_bounds!(e::SimplexEngine, iRows::AbstractVector{Int},
    lowers::AbstractVector{Float64}, uppers::AbstractVector{Float64})
    (length(iRows) == length(lowers) == length(uppers)) ||
        throw(ArgumentError("dimension mismatch in change_rows_bounds!"))
    for k ∈ eachindex(iRows)
        iRow = iRows[k]
        (1 <= iRow <= e.lp.num_row) ||
            throw(ArgumentError("row $iRow out of bounds"))
        e.lp.row_lower[iRow] = lowers[k]
        e.lp.row_upper[iRow] = uppers[k]
    end
    update_status!(e, kLpActionNewBounds)
    return e
end

"""
    change_cols_cost!(e::SimplexEngine, iCols, costs)

Update objective cost coefficients for a collection of columns in-place.
"""
function change_cols_cost!(e::SimplexEngine, iCols::AbstractVector{Int},
    costs::AbstractVector{Float64})
    length(iCols) == length(costs) ||
        throw(ArgumentError("dimension mismatch in change_cols_cost!"))
    for k ∈ eachindex(iCols)
        iCol = iCols[k]
        (1 <= iCol <= e.lp.num_col) ||
            throw(ArgumentError("column $iCol out of bounds"))
        e.lp.col_cost[iCol] = costs[k]
    end
    update_status!(e, kLpActionNewCosts)
    return e
end

"""`Highs_changeObjectiveSense`."""
function change_objective_sense!(e::SimplexEngine, sense::ObjSense)
    e.lp.sense == sense && return e
    e.lp.sense = sense
    update_status!(e, kLpActionNewCosts)
    return e
end

"""
    change_coeff!(e, iRow, iCol, value)

`Highs_changeCoeff`: modifies a constraint matrix coefficient (CSC format).
A coefficient magnitude below or equal to `small_matrix_value` is treated as zero:
it removes the existing entry, and a missing entry is not inserted. A true change
invalidates the entire state (`HEkk::updateStatus(kNewRows)` → `clear`).
"""
function change_coeff!(e::SimplexEngine, iRow::Int, iCol::Int, value::Float64)
    lp = e.lp
    (1 <= iRow <= lp.num_row && 1 <= iCol <= lp.num_col) ||
        throw(ArgumentError("coefficient ($iRow, $iCol) out of LP bounds"))
    m = lp.a_matrix
    is_colwise(m) || error("change_coeff! requires column-wise matrix")
    zero_new_value = abs(value) <= e.options.small_matrix_value
    change_el = 0
    for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
        if m.index[iEl] == iRow
            change_el = iEl
            break
        end
    end
    if change_el == 0
        # No existing non-zero: small coefficient is ignored.
        if !zero_new_value
            insert!(m.index, m.start[iCol + 1], iRow)
            insert!(m.value, m.start[iCol + 1], value)
            for i ∈ (iCol + 1):(lp.num_col + 1)
                m.start[i] += 1
            end
        end
    elseif zero_new_value
        # Coefficient zeroes an existing non-zero: remove it.
        deleteat!(m.index, change_el)
        deleteat!(m.value, change_el)
        for i ∈ (iCol + 1):(lp.num_col + 1)
            m.start[i] -= 1
        end
    else
        m.index[change_el] = iRow
        m.value[change_el] = value
    end
    # C++ source invalidates status even if coefficient did not change.
    update_status!(e, kLpActionNewRows)
    return e
end

"""
`HEkk::initialiseSimplexLpRandomVectors`: pseudo-random index permutations
and `numTotRandomValue`. Consumed in the exact order of the C++ source
(columns, then all variables, then fractions): determines the scan order for CHUZR.
"""
function initialise_simplex_lp_random_vectors!(e::SimplexEngine)
    num_col = e.lp.num_col
    num_tot = num_col + e.lp.num_row
    num_tot == 0 && return e
    if num_col > 0
        num_col_permutation = e.info.numColPermutation
        resize!(num_col_permutation, num_col)
        for i ∈ 1:num_col
            num_col_permutation[i] = i
        end
        shuffle!(e.random, num_col_permutation)
    end
    num_tot_permutation = e.info.numTotPermutation
    resize!(num_tot_permutation, num_tot)
    for i ∈ 1:num_tot
        num_tot_permutation[i] = i
    end
    shuffle!(e.random, num_tot_permutation)
    num_tot_random_value = e.info.numTotRandomValue
    resize!(num_tot_random_value, num_tot)
    for i ∈ 1:num_tot
        num_tot_random_value[i] = fraction(e.random)
    end
    return e
end

"""
Nonbasic move deduced solely from bounds — shared core of
`HEkk::setBasis` and `HEkk::setNonbasicMove`.
All bound combinations are covered; `kIllegalMoveValue` is never returned.
"""
function nonbasic_move_from_bounds(lower::Float64, upper::Float64)
    if lower == upper
        return kNonbasicMoveZe                      # fixed
    elseif lower > -kHighsInf
        if upper < kHighsInf
            # Boxed: bound closest to zero (C++ compares |lower| < |upper|).
            return abs(lower) < abs(upper) ? kNonbasicMoveUp : kNonbasicMoveDn
        end
        return kNonbasicMoveUp                      # lower bounded
    elseif upper < kHighsInf
        return kNonbasicMoveDn                      # upper bounded
    end
    return kNonbasicMoveZe                          # free
end

"""
    set_basis!(e)

Logical basis (`HEkk::setBasis`): nonbasic columns, basic slacks
(`basicIndex[iRow] = num_col + iRow`).
"""
function set_basis!(e::SimplexEngine)
    lp, basis = e.lp, e.basis
    setup!(basis, lp.num_col, lp.num_row)
    for iCol ∈ 1:lp.num_col
        basis.nonbasicFlag[iCol] = kNonbasicFlagTrue
        basis.nonbasicMove[iCol] = nonbasic_move_from_bounds(lp.col_lower[iCol],
            lp.col_upper[iCol])
    end
    for iRow ∈ 1:lp.num_row
        iVar = lp.num_col + iRow
        basis.nonbasicFlag[iVar] = kNonbasicFlagFalse
        basis.basicIndex[iRow] = iVar
        basis.hash = sparse_combine(basis.hash, iVar - 1)
    end
    e.info.num_basic_logicals = lp.num_row
    e.status.has_basis = true
    return e
end

"""
    handle_rank_deficiency!(e)

`HEkk::handleRankDeficiency`: factor has completed a singular basis
with logical variables (`buildHandleRankDeficiency`/`buildMarkSingC`);
flags are synchronized, the change is marked taboo (`kSingular`),
and the row-wise view is invalidated.
"""
function handle_rank_deficiency!(e::SimplexEngine)
    factor = e.nla.factor
    for k ∈ 1:factor.rank_deficiency
        row_in = factor.row_with_no_pivot[k]
        variable_in = e.lp.num_col + row_in
        variable_out = factor.var_with_no_pivot[k]
        e.basis.nonbasicFlag[variable_in] = kNonbasicFlagFalse
        variable_out > 0 &&
            (e.basis.nonbasicFlag[variable_out] = kNonbasicFlagTrue)
        add_bad_basis_change!(e, row_in, variable_in, variable_out,
            kBadBasisChangeSingular, true)
    end
    e.status.has_ar_matrix = false
    e.status.has_dual_steepest_edge_weights = false
    return e
end

"""`HEkk::logicalBasis` — all basic variables are logical (slack) variables."""
function logical_basis(e::SimplexEngine)
    return all(iRow -> e.basis.basicIndex[iRow] > e.lp.num_col,
        1:e.lp.num_row)
end

"""
`HEkk::setBasis(const HighsBasis&)`: installs `from` into the shared basis
(in-place copy, `basicIndex` remains the buffer of the factor and NLA).
"""
function restore_basis!(e::SimplexEngine, from::SimplexBasis)
    basis = e.basis
    length(basis.basicIndex) == length(from.basicIndex) ||
        throw(ArgumentError("basis sizes mismatch"))
    copyto!(basis.basicIndex, from.basicIndex)
    copyto!(basis.nonbasicFlag, from.nonbasicFlag)
    copyto!(basis.nonbasicMove, from.nonbasicMove)
    basis.hash = from.hash
    e.status.has_basis = true
    return e
end

"""
`HApp::solveLpSimplex` (post-solve): `basis_ = getHighsBasis(ekk)` —
returned basis becomes the cached basis of the Highs level, used on next
solve if the Ekk basis was invalidated.
"""
function store_solution_basis!(e::SimplexEngine)
    if e.status.has_basis
        copy_basis!(e.highs_basis, e.basis)
        e.highs_basis_valid = true
    end
    return e
end

"""`HEkk::clearRayRecords` — clears primal ray record (`clear` of `HighsRayRecord`)."""
function clear_ray_records!(e::SimplexEngine)
    clear!(e.primal_ray_record)
    return e
end

"""Basis hash: combination of basic variable indices."""
function basis_hash(basis::SimplexBasis)
    hash = UInt64(0)
    for iVar ∈ eachindex(basis.nonbasicFlag)
        if basis.nonbasicFlag[iVar] == kNonbasicFlagFalse
            hash = sparse_combine(hash, iVar - 1)
        end
    end
    return hash
end

"""
    set_nonbasic_move!(e)

`HEkk::setNonbasicMove`: recomputes `nonbasicMove` for all variables from
the **LP** bounds (not the workspace bounds), without modifying values.
A basic variable receives `kNonbasicMoveZe`.
"""
function set_nonbasic_move!(e::SimplexEngine)
    lp, basis = e.lp, e.basis
    num_col = lp.num_col
    @inbounds for iVar ∈ 1:(num_col + lp.num_row)
        if basis.nonbasicFlag[iVar] == kNonbasicFlagFalse
            basis.nonbasicMove[iVar] = kNonbasicMoveZe
            continue
        end
        if iVar <= num_col
            lower, upper = lp.col_lower[iVar], lp.col_upper[iVar]
        else
            iRow = iVar - num_col
            lower, upper = -lp.row_upper[iRow], -lp.row_lower[iRow]
        end
        basis.nonbasicMove[iVar] = nonbasic_move_from_bounds(lower, upper)
    end
    return e
end

"""`HEkk::initialiseLpColBound` — workspace column bounds."""
function initialise_lp_col_bound!(e::SimplexEngine)
    lp, info = e.lp, e.info
    @inbounds for iCol ∈ 1:lp.num_col
        info.workLower[iCol] = lp.col_lower[iCol]
        info.workUpper[iCol] = lp.col_upper[iCol]
        info.workRange[iCol] = info.workUpper[iCol] - info.workLower[iCol]
        info.workLowerShift[iCol] = 0.0
        info.workUpperShift[iCol] = 0.0
    end
    return e
end

"""`HEkk::initialiseLpRowBound` — logical bounds `[-row_upper, -row_lower]`."""
function initialise_lp_row_bound!(e::SimplexEngine)
    lp, info = e.lp, e.info
    num_col = lp.num_col
    @inbounds for iRow ∈ 1:lp.num_row
        iVar = num_col + iRow
        info.workLower[iVar] = -lp.row_upper[iRow]
        info.workUpper[iVar] = -lp.row_lower[iRow]
        info.workRange[iVar] = info.workUpper[iVar] - info.workLower[iVar]
        info.workLowerShift[iVar] = 0.0
        info.workUpperShift[iVar] = 0.0
    end
    return e
end

"""
    initialise_bound!(e, algorithm, solve_phase; perturb = false)

`HEkk::initialiseBound`: workspace bounds from LP, followed by (for dual simplex
outside phase 2) special bounds `[-1000, 1000]`/`[-1, 0]`/`[0, 1]`/`[0, 0]` making
the dual objective equal to minus the sum of infeasibilities. For primal simplex,
random perturbation of bounds (base `multiplier * 5e-7`, relative to bound,
absolute if `|bound| < 1`) expands finite bounds; fixed nonbasic variables are kept intact.
"""
function initialise_bound!(e::SimplexEngine, algorithm::SimplexAlgorithm,
    solve_phase::Int; perturb::Bool=false)
    initialise_lp_col_bound!(e)
    initialise_lp_row_bound!(e)
    info = e.info
    info.bounds_shifted = false
    info.bounds_perturbed = false
    if algorithm == kPrimal
        (!perturb || info.primal_simplex_bound_perturbation_multiplier == 0) &&
            return e
        base = info.primal_simplex_bound_perturbation_multiplier * 5e-7
        for iVar ∈ 1:(e.lp.num_col + e.lp.num_row)
            lower = info.workLower[iVar]
            upper = info.workUpper[iVar]
            # Fixed nonbasic variable remains at bound: not perturbed
            if e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue && lower == upper
                continue
            end
            random_value = info.numTotRandomValue[iVar]
            if lower > -kHighsInf
                if lower < -1
                    lower -= random_value * base * (-lower)
                elseif lower < 1
                    lower -= random_value * base
                else
                    lower -= random_value * base * lower
                end
                info.workLower[iVar] = lower
            end
            if upper < kHighsInf
                if upper < -1
                    upper += random_value * base * (-upper)
                elseif upper < 1
                    upper += random_value * base
                else
                    upper += random_value * base * upper
                end
                info.workUpper[iVar] = upper
            end
            info.workRange[iVar] = info.workUpper[iVar] - info.workLower[iVar]
            if e.basis.nonbasicFlag[iVar] == kNonbasicFlagFalse
                continue
            end
            if e.basis.nonbasicMove[iVar] > 0
                info.workValue[iVar] = lower
            elseif e.basis.nonbasicMove[iVar] < 0
                info.workValue[iVar] = upper
            end
        end
        for iRow ∈ 1:e.lp.num_row
            iVar = e.basis.basicIndex[iRow]
            info.baseLower[iRow] = info.workLower[iVar]
            info.baseUpper[iRow] = info.workUpper[iVar]
        end
        info.bounds_perturbed = true
        return e
    end
    solve_phase == kSolvePhase2 && return e
    for iVar ∈ 1:(e.lp.num_col + e.lp.num_row)
        if info.workLower[iVar] == -kHighsInf && info.workUpper[iVar] == kHighsInf
            info.workLower[iVar] = -1000.0          # free
            info.workUpper[iVar] = 1000.0
        elseif info.workLower[iVar] == -kHighsInf
            info.workLower[iVar] = -1.0             # upper bounded
            info.workUpper[iVar] = 0.0
        elseif info.workUpper[iVar] == kHighsInf
            info.workLower[iVar] = 0.0              # lower bounded
            info.workUpper[iVar] = 1.0
        else
            info.workLower[iVar] = 0.0              # boxed or fixed
            info.workUpper[iVar] = 0.0
        end
        info.workRange[iVar] = info.workUpper[iVar] - info.workLower[iVar]
    end
    return e
end

"""
`HEkk::initialiseLpColCost` — signed costs (`sense`) scaled by
`2^cost_scale_factor`; shifts are reset to zero.
"""
function initialise_lp_col_cost!(e::SimplexEngine)
    cost_scale_factor = 2.0^e.options.cost_scale_factor
    sense_scaled = Int(e.lp.sense) * cost_scale_factor
    col_cost = e.lp.col_cost
    workCost = e.info.workCost
    workShift = e.info.workShift
    @inbounds for iCol ∈ 1:e.lp.num_col
        workCost[iCol] = sense_scaled * col_cost[iCol]
        workShift[iCol] = 0.0
    end
    return e
end

"""`HEkk::initialiseLpRowCost` — zero cost for logical (slack) variables."""
function initialise_lp_row_cost!(e::SimplexEngine)
    workCost = e.info.workCost
    workShift = e.info.workShift
    @inbounds for iVar ∈ (e.lp.num_col + 1):(e.lp.num_col + e.lp.num_row)
        workCost[iVar] = 0.0
        workShift[iVar] = 0.0
    end
    return e
end

"""
    initialise_cost!(e, algorithm, solve_phase; perturb = false)

`HEkk::initialiseCost`: copies LP costs (`solve_phase` does not affect computation,
matching C++ source), and if `perturb` is requested with non-zero multiplier,
applies dual random perturbation: `xpert = (1 + r_i) (|c_i| + 1) * base` where
`base = multiplier * 5e-7 * max|c|` (reduced by `sqrt(sqrt(·))` beyond 100, and capped
at 1 if fewer than 1% of variables are boxed); slack costs receive
`(0.5 - r_i) * multiplier * 1e-12`.
"""
function initialise_cost!(e::SimplexEngine, algorithm::SimplexAlgorithm,
    solve_phase::Int; perturb::Bool=false)
    initialise_lp_col_cost!(e)
    initialise_lp_row_cost!(e)
    info = e.info
    info.costs_shifted = false
    info.costs_perturbed = false
    algorithm == kPrimal && return e
    multiplier = info.dual_simplex_cost_perturbation_multiplier
    (!perturb || multiplier == 0) && return e
    max_abs_cost = 0.0
    for i ∈ 1:e.lp.num_col
        max_abs_cost = max(max_abs_cost, abs(info.workCost[i]))
    end
    if max_abs_cost > 100.0
        max_abs_cost = sqrt(sqrt(max_abs_cost))
    end
    num_tot = e.lp.num_col + e.lp.num_row
    boxed_rate = 0.0
    for i ∈ 1:num_tot
        boxed_rate += info.workRange[i] < 1e30
    end
    boxed_rate /= num_tot
    if boxed_rate < 0.01
        max_abs_cost = min(max_abs_cost, 1.0)
    end
    cost_perturbation_base = multiplier * 5e-7 * max_abs_cost
    for i ∈ 1:e.lp.num_col
        lower = e.lp.col_lower[i]
        upper = e.lp.col_upper[i]
        xpert = (1 + info.numTotRandomValue[i]) *
                (abs(info.workCost[i]) + 1) * cost_perturbation_base
        if lower == -kHighsInf && upper == kHighsInf
            # free: no perturbation
        elseif upper == kHighsInf
            info.workCost[i] += xpert                  # lower bounded
        elseif lower == -kHighsInf
            info.workCost[i] -= xpert                  # upper bounded
        elseif lower != upper
            info.workCost[i] += info.workCost[i] >= 0 ? xpert : -xpert  # boxed
        end
    end
    row_cost_perturbation_base = multiplier * 1e-12
    for i ∈ (e.lp.num_col + 1):num_tot
        info.workCost[i] +=
            (0.5 - info.numTotRandomValue[i]) * row_cost_perturbation_base
    end
    info.costs_perturbed = true
    return e
end

"""
    initialise_nonbasic_value_and_move!(e)

`HEkk::initialiseNonbasicValueAndMove`: sets nonbasic values and moves
from `nonbasicFlag` and workspace bounds. Boxed variables keep the side
designated by their original `nonbasicMove`; invalid move is defaulted
to `kNonbasicMoveUp`. Basics remain zero: their values are assigned by `compute_primal!`.
"""
function initialise_nonbasic_value_and_move!(e::SimplexEngine)
    basis, info = e.basis, e.info
    for iVar ∈ 1:(e.lp.num_col + e.lp.num_row)
        if basis.nonbasicFlag[iVar] == kNonbasicFlagFalse
            basis.nonbasicMove[iVar] = kNonbasicMoveZe
            continue
        end
        lower, upper = info.workLower[iVar], info.workUpper[iVar]
        original_move = basis.nonbasicMove[iVar]
        if lower == upper
            value, move = lower, kNonbasicMoveZe
        elseif lower > -kHighsInf
            if upper < kHighsInf
                if original_move == kNonbasicMoveUp
                    value, move = lower, kNonbasicMoveUp
                elseif original_move == kNonbasicMoveDn
                    value, move = upper, kNonbasicMoveDn
                else
                    value, move = lower, kNonbasicMoveUp
                end
            else
                value, move = lower, kNonbasicMoveUp
            end
        elseif upper < kHighsInf
            value, move = upper, kNonbasicMoveDn
        else
            value, move = 0.0, kNonbasicMoveZe
        end
        basis.nonbasicMove[iVar] = move
        info.workValue[iVar] = value
    end
    return e
end

"""`HEkk::updateOperationResultDensity` — exponential moving average of densities."""
update_operation_result_density(density::Float64, local_density::Float64) =
    (1 - kRunningAverageMultiplier) * density +
    kRunningAverageMultiplier * local_density

"""`HEkk::fullBtran` — full BTRAN, then updates moving average `dual_col_density`."""
function full_btran!(e::SimplexEngine, buffer::HVector)
    btran!(e.nla, buffer, e.info.dual_col_density)
    e.info.dual_col_density = update_operation_result_density(
        e.info.dual_col_density, buffer.count / e.lp.num_row)
    return buffer
end

"""`HEkk::fullPrice` — `full_row = A^T full_col`, column-by-column pricing."""
function full_price!(e::SimplexEngine, full_col::HVector, full_row::HVector)
    clear!(full_row)
    price_by_column!(e.lp.a_matrix, full_row, full_col)
    return full_row
end

"""
    compute_primal!(e)

`HEkk::computePrimal`: `x_B = -B^{-1} N x_N` via `collectAj` followed by FTRAN.
`baseValue` follows basic column ordering, `baseLower`/`baseUpper` are synchronized,
and primal infeasibility counters are invalidated.
"""
function compute_primal!(e::SimplexEngine)
    num_row, num_col = e.lp.num_row, e.lp.num_col
    primal_col = e.primal_col
    clear!(primal_col)
    @inbounds for iVar ∈ 1:(num_col + num_row)
        if e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue &&
           e.info.workValue[iVar] != 0.0
            collect_aj!(e.lp.a_matrix, primal_col, iVar, e.info.workValue[iVar])
        end
    end
    if primal_col.count != 0
        ftran!(e.nla, primal_col, e.info.primal_col_density)
        e.info.primal_col_density = update_operation_result_density(
            e.info.primal_col_density, primal_col.count / num_row)
    end
    @inbounds @simd for iRow ∈ 1:num_row
        e.info.baseValue[iRow] = -primal_col.array[iRow]
    end
    @inbounds for iRow ∈ 1:num_row
        iVar = e.basis.basicIndex[iRow]
        e.info.baseLower[iRow] = e.info.workLower[iVar]
        e.info.baseUpper[iRow] = e.info.workUpper[iVar]
    end
    e.info.num_primal_infeasibilities = kHighsIllegalInfeasibilityCount
    e.info.max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure
    e.info.sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure
    return e
end

"""
    compute_dual!(e)

`HEkk::computeDual`: `workDual = workCost + workShift - [A I]^T pi` with
`pi = B^{-T} c_B` (full BTRAN on basic costs), `A^T pi` via `priceByColumn`.
Basic dual values are zero within precision. Dual infeasibility counters are invalidated.
"""
function compute_dual!(e::SimplexEngine)
    num_col, num_row = e.lp.num_col, e.lp.num_row
    num_tot = num_col + num_row
    dual_col = e.dual_col
    clear!(dual_col)
    @inbounds for iRow ∈ 1:num_row
        iVar = e.basis.basicIndex[iRow]
        value = e.info.workCost[iVar] + e.info.workShift[iVar]
        if value != 0.0
            dual_col.count += 1
            dual_col.index[dual_col.count] = iRow
            dual_col.array[iRow] = value
        end
    end
    @inbounds @simd for iVar ∈ 1:num_tot
        e.info.workDual[iVar] = e.info.workCost[iVar] + e.info.workShift[iVar]
    end
    if dual_col.count != 0
        full_btran!(e, dual_col)
        dual_row = e.dual_row
        full_price!(e, dual_col, dual_row)
        @inbounds @simd for iCol ∈ 1:num_col
            e.info.workDual[iCol] -= dual_row.array[iCol]
        end
        @inbounds @simd for iVar ∈ (num_col + 1):num_tot
            e.info.workDual[iVar] -= dual_col.array[iVar - num_col]
        end
    end
    e.info.num_dual_infeasibilities = kHighsIllegalInfeasibilityCount
    e.info.max_dual_infeasibility = kHighsIllegalInfeasibilityMeasure
    e.info.sum_dual_infeasibilities = kHighsIllegalInfeasibilityMeasure
    return e
end

"""
    compute_primal_objective_value!(e)

`HEkk::computePrimalObjectiveValue`:
`cost_scale * (c_B^T x_B + c_N^T x_N) + offset`. Original costs are used
(not `workCost`) and only structural variables contribute.
Returns the value, also stored in `info.primal_objective_value`.
"""
function compute_primal_objective_value!(e::SimplexEngine)
    value = 0.0
    @inbounds for iRow ∈ 1:e.lp.num_row
        iVar = e.basis.basicIndex[iRow]
        if iVar <= e.lp.num_col
            value += e.info.baseValue[iRow] * e.lp.col_cost[iVar]
        end
    end
    @inbounds for iCol ∈ 1:e.lp.num_col
        if e.basis.nonbasicFlag[iCol] == kNonbasicFlagTrue
            value += e.info.workValue[iCol] * e.lp.col_cost[iCol]
        end
    end
    value *= e.cost_scale
    value += e.lp.offset
    e.info.primal_objective_value = value
    e.status.has_primal_objective_value = true
    return value
end

"""
    compute_dual_objective_value!(e, phase = kSolvePhase2)

`HEkk::computeDualObjectiveValue`: `cost_scale * Σ x_i workDual_i` over nonbasics,
plus `sense * offset` except in phase 1 ("dual objective has no offset").
Returns the value, also stored in `info.dual_objective_value`.
"""
function compute_dual_objective_value!(e::SimplexEngine,
    phase::Int=kSolvePhase2)
    value = 0.0
    @inbounds for iVar ∈ 1:(e.lp.num_col + e.lp.num_row)
        if e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue
            value += e.info.workValue[iVar] * e.info.workDual[iVar]
        end
    end
    value *= e.cost_scale
    if phase != kSolvePhase1
        value += Int(e.lp.sense) * e.lp.offset
    end
    e.info.dual_objective_value = value
    e.status.has_dual_objective_value = true
    return value
end

# --- Dual simplex primitives from `HEkk` ------------------------------------

"""
    compute_factor!(e)

`HEkk::computeFactor`: computes basis INVERT on the current basis.
Returns rank deficiency (`0` if B^{-1} is non-singular). `update_count` is reset
and `build_synthetic_tick` is recorded (used for triggering reinversions).
"""
function compute_factor!(e::SimplexEngine)
    rank_deficiency = invert!(e.nla)
    e.build_synthetic_tick = e.nla.build_synthetic_tick
    if rank_deficiency != 0
        e.status.has_invert = false
        e.status.has_fresh_invert = false
    else
        e.status.has_invert = true
        e.status.has_fresh_invert = true
    end
    e.info.update_count = 0
    return rank_deficiency
end

"""
    get_nonsingular_inverse!(e, solve_phase)

`HEkk::getNonsingularInverse` with backtracking: returns `false` if the basis is
singular and cannot be restored from a valid backtracking basis.
"""
function get_nonsingular_inverse!(e::SimplexEngine, solve_phase::Int)
    # DSE weights are indexed by rows: scatter them according to `basic_index`
    # before INVERT, then gather according to resulting permutation.
    basic_index = e.basis.basicIndex
    basic_index_before = e.basic_index_before
    copyto!(basic_index_before, basic_index)
    simplex_update_count = e.info.update_count
    scattered = e.scattered_dual_edge_weight
    for i ∈ 1:e.lp.num_row
        scattered[basic_index[i]] = e.dual_edge_weight[i]
    end
    rank_deficiency = compute_factor!(e)
    if rank_deficiency != 0
        # Singular basis: backtrack to last non-singular basis.
        deficient_hash = e.basis.hash
        get_backtracking_basis!(e) || return false
        e.info.backtracking = true
        empty!(e.visited_basis)
        push!(e.visited_basis, e.basis.hash, deficient_hash)
        e.status.has_ar_matrix = false
        e.status.has_fresh_rebuild = false
        e.status.has_dual_objective_value = false
        e.status.has_primal_objective_value = false
        compute_factor!(e) == 0 || return false
        simplex_update_count <= 1 && return false
        e.info.update_limit = simplex_update_count ÷ 2
    else
        put_backtracking_basis!(e, basic_index_before)
        e.info.backtracking = false
        e.info.update_limit = e.options.simplex_update_limit
    end
    for i ∈ 1:e.lp.num_row
        e.dual_edge_weight[i] = scattered[basic_index[i]]
    end
    return true
end

"""
    put_backtracking_basis!(e)
    put_backtracking_basis!(e, basic_index_before_compute_factor)

`HEkk::putBacktrackingBasis`: saves current basis (or pre-INVERT basic index order)
and associated shift/weight state.
"""
function put_backtracking_basis!(e::SimplexEngine)
    info = e.info
    info.valid_backtracking_basis = true
    copy_basis!(info.backtracking_basis, e.basis)
    info.backtracking_basis_costs_shifted = info.costs_shifted
    info.backtracking_basis_costs_perturbed = info.costs_perturbed
    info.backtracking_basis_bounds_shifted = info.bounds_shifted
    info.backtracking_basis_bounds_perturbed = info.bounds_perturbed
    resize!(info.backtracking_basis_workShift, length(info.workShift))
    copyto!(info.backtracking_basis_workShift, info.workShift)
    resize!(info.backtracking_basis_workLowerShift, length(info.workLowerShift))
    copyto!(info.backtracking_basis_workLowerShift, info.workLowerShift)
    resize!(info.backtracking_basis_workUpperShift, length(info.workUpperShift))
    copyto!(info.backtracking_basis_workUpperShift, info.workUpperShift)
    resize!(info.backtracking_basis_edge_weight, length(e.scattered_dual_edge_weight))
    copyto!(info.backtracking_basis_edge_weight, e.scattered_dual_edge_weight)
    return e
end

function put_backtracking_basis!(e::SimplexEngine,
    basic_index_before_compute_factor::Vector{Int})
    put_backtracking_basis!(e)
    bb = e.info.backtracking_basis
    resize!(bb.basicIndex, length(basic_index_before_compute_factor))
    copyto!(bb.basicIndex, basic_index_before_compute_factor)
    return e
end

"""`HEkk::getBacktrackingBasis` — restores the last non-singular basis."""
function get_backtracking_basis!(e::SimplexEngine)
    info = e.info
    info.valid_backtracking_basis || return false
    copy_basis!(e.basis, info.backtracking_basis)
    info.costs_shifted = info.backtracking_basis_costs_shifted
    info.costs_perturbed = info.backtracking_basis_costs_perturbed
    info.bounds_shifted = info.backtracking_basis_bounds_shifted
    info.bounds_perturbed = info.backtracking_basis_bounds_perturbed
    resize!(info.workShift, length(info.backtracking_basis_workShift))
    copyto!(info.workShift, info.backtracking_basis_workShift)
    copyto!(e.scattered_dual_edge_weight, info.backtracking_basis_edge_weight)
    return true
end

"""
    is_bad_basis_change!(e, algorithm, variable_in, row_out, rebuild_reason)

`HEkk::isBadBasisChange`: detects cycling (basis hash visited on successive iterations)
or a basis change already recorded as bad, and marks it taboo.
"""
function is_bad_basis_change!(e::SimplexEngine, algorithm::SimplexAlgorithm,
    variable_in::Int, row_out::Int, rebuild_reason::Int)
    rebuild_reason != kRebuildReasonNo && return false
    (variable_in == -1 || row_out == -1) && return false
    currhash = e.basis.hash
    variable_out = e.basis.basicIndex[row_out]
    currhash = sparse_inverse_combine(currhash, variable_out - 1)
    currhash = sparse_combine(currhash, variable_in - 1)
    cycling_detected = false
    possible_cycling = currhash ∈ e.visited_basis
    if possible_cycling
        if e.iteration_count == e.previous_iteration_cycling_detected + 1
            cycling_detected = true
        else
            e.previous_iteration_cycling_detected = e.iteration_count
        end
    end
    if cycling_detected
        add_bad_basis_change!(e, row_out, variable_out, variable_in,
            kBadBasisChangeCycling, true)
        return true
    end
    for change ∈ e.bad_basis_change
        if change.variable_out == variable_out &&
           change.variable_in == variable_in && change.row_out == row_out
            change.taboo = true
            return true
        end
    end
    return false
end

"""
    add_bad_basis_change!(e, row_out, variable_out, variable_in, reason, taboo)

`HEkk::addBadBasisChange`: records (or updates taboo flag of) a bad basis change.
Returns its index.
"""
function add_bad_basis_change!(e::SimplexEngine, row_out::Int,
    variable_out::Int, variable_in::Int, reason::Int, taboo::Bool)
    num_bad = length(e.bad_basis_change)
    index = -1
    for i ∈ 1:num_bad
        record = e.bad_basis_change[i]
        if record.row_out == row_out && record.variable_out == variable_out &&
           record.variable_in == variable_in && record.reason == reason
            index = i
            break
        end
    end
    if index < 0
        push!(e.bad_basis_change,
            BadBasisChange(row_out, variable_out, variable_in, reason, taboo))
        index = length(e.bad_basis_change)
    else
        e.bad_basis_change[index].taboo = taboo
    end
    return index
end

"""`HEkk::clearBadBasisChange` — empties list, or clears only a specific reason."""
function clear_bad_basis_change!(e::SimplexEngine, reason::Int)
    if reason == kBadBasisChangeAll
        empty!(e.bad_basis_change)
    else
        filter!(change -> change.reason != reason, e.bad_basis_change)
    end
    return e
end

"""`HEkk::clearBadBasisChangeTabooFlag`."""
function clear_bad_basis_change_taboo_flag!(e::SimplexEngine)
    for change ∈ e.bad_basis_change
        change.taboo = false
    end
    return e
end

"""`HEkk::tabooBadBasisChange` — at least one taboo change is recorded."""
taboo_bad_basis_change(e::SimplexEngine) =
    any(change -> change.taboo, e.bad_basis_change)

"""`HEkk::applyTabooRowOut` — zeroes out infeasibility of taboo rows."""
function apply_taboo_row_out!(e::SimplexEngine, values::Vector{Float64},
    overwrite_with::Float64)
    for change ∈ e.bad_basis_change
        if change.taboo
            iRow = change.row_out
            change.save_value = values[iRow]
            values[iRow] = overwrite_with
        end
    end
    return e
end

"""`HEkk::unapplyTabooRowOut` — restores row values in reverse order."""
function unapply_taboo_row_out!(e::SimplexEngine, values::Vector{Float64})
    for iX ∈ length(e.bad_basis_change):-1:1
        change = e.bad_basis_change[iX]
        if change.taboo
            values[change.row_out] = change.save_value
        end
    end
    return e
end

"""
`HEkk::updateBadBasisChange`: clears bad basis changes whose pivot entry
has a primal effect of at least feasibility tolerance (predicate in C++
source is `>= tolerance`, filtered via `remove_if`).
"""
function update_bad_basis_change!(e::SimplexEngine, col_aq::HVector,
    theta_primal::Float64)
    isempty(e.bad_basis_change) && return e
    tolerance = e.options.primal_feasibility_tolerance
    filter!(change ->
        abs(col_aq.array[change.row_out] * theta_primal) < tolerance,
        e.bad_basis_change)
    return e
end

"""
    initialise_control!(e)

`HEkk::initialiseControl`: DSE/Devex threshold, control counter,
densities reset to initial values (`dual_col_density = 1`).
"""
function initialise_control!(e::SimplexEngine)
    info = e.info
    info.allow_dual_steepest_edge_to_devex_switch =
        e.options.simplex_dual_edge_weight_strategy ==
        kSimplexEdgeWeightStrategyChoose
    info.control_iteration_count0 = e.iteration_count
    info.col_aq_density = 0.0
    info.row_ep_density = 0.0
    info.row_ap_density = 0.0
    info.row_DSE_density = 0.0
    info.col_steepest_edge_density = 0.0
    info.col_BFRT_density = 0.0
    info.primal_col_density = 0.0
    info.dual_col_density = 1.0
    info.col_basic_feasibility_change_density = 0.0
    info.row_basic_feasibility_change_density = 0.0
    info.costly_DSE_frequency = 0.0
    info.num_costly_DSE_iteration = 0
    info.costly_DSE_measure = 0.0
    info.average_log_low_DSE_weight_error = 0.0
    info.average_log_high_DSE_weight_error = 0.0
    return e
end

"""`HEkk::computeDualSteepestEdgeWeights` — DSE weights for all rows."""
function compute_dual_steepest_edge_weights!(e::SimplexEngine,
    initial::Bool=false)
    row_ep = e.dual_col
    for iRow ∈ 1:e.lp.num_row
        e.dual_edge_weight[iRow] =
            compute_dual_steepest_edge_weight(e, iRow, row_ep)
    end
    return e
end

"""
`HEkk::computeDualSteepestEdgeWeight`: `‖B^{-T} e_p‖²` in scaled space,
with exponential moving average for `row_ep_density`.
"""
function compute_dual_steepest_edge_weight(e::SimplexEngine, iRow::Int,
    row_ep::HVector)
    clear!(row_ep)
    row_ep.count = 1
    row_ep.index[1] = iRow
    row_ep.array[iRow] = 1.0
    row_ep.packFlag = false
    btran_in_scaled_space!(e.nla, row_ep, e.info.row_ep_density)
    e.info.row_ep_density = update_operation_result_density(
        e.info.row_ep_density, row_ep.count / e.lp.num_row)
    return norm2(row_ep)
end

"""
`HEkk::updateDualSteepestEdgeWeights`:
`w_i += a_i (w_p a_i + Kai y_i)`, clamped to `kMinDualSteepestEdgeWeight`.
"""
function update_dual_steepest_edge_weights!(e::SimplexEngine, row_out::Int,
    variable_in::Int, column::HVector, new_pivotal_edge_weight::Float64,
    kai::Float64, dual_steepest_edge_array::AbstractVector{Float64})
    num_row = e.lp.num_row
    col_aq_scale = variable_scale_factor(e.nla, variable_in)
    col_ap_scale = basic_col_scale_factor(e.nla, row_out)
    inv_col_ap_scale = 1.0 / col_ap_scale
    use_row_indices, to_entry = sparse_loop_style(column.count, num_row)
    @inbounds for iEntry ∈ 1:to_entry
        iRow = use_row_indices ? column.index[iEntry] : iEntry
        aa_iRow = column.array[iRow]
        aa_iRow == 0.0 && continue
        dual_steepest_edge_array_value = dual_steepest_edge_array[iRow]
        # `convert_to_scaled_space = !simplex_in_scaled_space_`: scaling conversion
        # in `HEkk::updateDualSteepestEdgeWeights` is identity without scaling factors,
        # and skipped when LP is already scaled.
        if !e.lp.is_scaled
            aa_iRow /= basic_col_scale_factor(e.nla, iRow)
            aa_iRow *= col_aq_scale
            dual_steepest_edge_array_value *= inv_col_ap_scale
        end
        e.dual_edge_weight[iRow] +=
            aa_iRow * (new_pivotal_edge_weight * aa_iRow +
                       kai * dual_steepest_edge_array_value)
        e.dual_edge_weight[iRow] = max(kMinDualSteepestEdgeWeight,
            e.dual_edge_weight[iRow])
    end
    return e
end

"""
`HEkk::updateDualDevexWeights`: `w_i = max(w_i, w_p a_i²)` over listed
entries of the pivot column.
"""
function update_dual_devex_weights!(e::SimplexEngine, column::HVector,
    new_pivotal_edge_weight::Float64)
    use_row_indices, to_entry = sparse_loop_style(column.count, e.lp.num_row)
    @inbounds for iEntry ∈ 1:to_entry
        iRow = use_row_indices ? column.index[iEntry] : iEntry
        aa_iRow = column.array[iRow]
        e.dual_edge_weight[iRow] = max(e.dual_edge_weight[iRow],
            new_pivotal_edge_weight * aa_iRow * aa_iRow)
    end
    return e
end

"""`HEkk::assessDSEWeightError` — exponential moving average of weight errors."""
function assess_dse_weight_error!(e::SimplexEngine,
    computed_edge_weight::Float64, updated_edge_weight::Float64)
    if updated_edge_weight < computed_edge_weight
        relative_deviation = computed_edge_weight / updated_edge_weight
        e.info.average_log_low_DSE_weight_error =
            0.99 * e.info.average_log_low_DSE_weight_error +
            0.01 * log(relative_deviation)
    else
        relative_deviation = updated_edge_weight / computed_edge_weight
        e.info.average_log_high_DSE_weight_error =
            0.99 * e.info.average_log_high_DSE_weight_error +
            0.01 * log(relative_deviation)
    end
    return e
end

"""`HEkk::resetSyntheticClock` — resets synthetic clock tick after INVERT."""
function reset_synthetic_clock!(e::SimplexEngine)
    e.build_synthetic_tick = e.nla.build_synthetic_tick
    e.total_synthetic_tick = 0.0
    return e
end

"""
    rebuild_refactor(e, rebuild_reason)

`HEkk::rebuildRefactor`: with `no_unnecessary_rebuild_refactor` (default),
reinversion is only executed if factor error on test system exceeds
`rebuild_refactor_solution_error_tolerance`.
"""
function rebuild_refactor(e::SimplexEngine, rebuild_reason::Int)
    e.info.update_count == 0 && return false
    refactor = true
    if e.options.no_unnecessary_rebuild_refactor
        if rebuild_reason == kRebuildReasonNo ||
           rebuild_reason == kRebuildReasonPossiblyOptimal ||
           rebuild_reason == kRebuildReasonPossiblyPhase1Feasible ||
           rebuild_reason == kRebuildReasonPossiblyPrimalUnbounded ||
           rebuild_reason == kRebuildReasonPossiblyDualUnbounded ||
           rebuild_reason == kRebuildReasonPrimalInfeasibleInPrimalSimplex
            refactor = false
            error_tolerance = e.options.rebuild_refactor_solution_error_tolerance
            if error_tolerance > 0
                solution_error = factor_solve_error(e)
                refactor = solution_error > error_tolerance
            end
        end
    end
    return refactor
end

"""
    factor_solve_error(e)

`HEkk::factorSolveError`: builds a random test solution with at most 50 non-zeros
(RNG seed 1), solves corresponding systems, and measures max residual.
Used to decide whether reinversion is necessary.
"""
function factor_solve_error(e::SimplexEngine)
    num_col = e.lp.num_col
    num_row = e.lp.num_row
    a_matrix = e.lp.a_matrix
    ar_matrix = e.ar_matrix
    basic_index = e.basis.basicIndex
    random = initialise!(e.fse_random, 1)
    ftran_rhs = e.primal_col
    clear!(ftran_rhs)
    solution_num_nz = min(50, (num_row + 1) ÷ 2)
    solution_value = e.fse_solution_value
    solution_index = e.fse_solution_index
    solution_nonzero = e.fse_solution_nonzero
    fill!(solution_nonzero, false)
    sol_count = 0
    @inbounds while true
        iRow = integer(random, num_row) + 1
        solution_nonzero[iRow] && continue
        value = fraction(random)
        sol_count += 1
        solution_value[sol_count] = value
        solution_index[sol_count] = iRow
        solution_nonzero[iRow] = true
        collect_aj!(a_matrix, ftran_rhs, basic_index[iRow], value)
        sol_count == solution_num_nz && break
    end
    # BTRAN: (B^T x) restricted to basic columns, via row-wise view.
    btran_scattered_rhs = e.fse_btran_scattered
    fill!(btran_scattered_rhs, 0.0)
    @inbounds for iX ∈ 1:solution_num_nz
        iRow = solution_index[iX]
        val_X = solution_value[iX]
        for iEl ∈ ar_matrix.p_end[iRow]:(ar_matrix.start[iRow + 1] - 1)
            iCol = ar_matrix.index[iEl]
            btran_scattered_rhs[iCol] +=
                ar_matrix.value[iEl] * val_X
        end
        iCol = num_col + iRow
        if e.basis.nonbasicFlag[iCol] == kNonbasicFlagFalse
            btran_scattered_rhs[iCol] = val_X
        end
    end
    btran_rhs = e.dual_col
    clear!(btran_rhs)
    @inbounds for iRow ∈ 1:num_row
        iCol = basic_index[iRow]
        if btran_scattered_rhs[iCol] != 0.0
            btran_rhs.count += 1
            btran_rhs.array[iRow] = btran_scattered_rhs[iCol]
            btran_rhs.index[btran_rhs.count] = iRow
        end
    end
    expected_density = solution_num_nz * e.info.col_aq_density
    ftran!(e.nla, ftran_rhs, expected_density)
    btran!(e.nla, btran_rhs, expected_density)
    ftran_solution_error = 0.0
    btran_solution_error = 0.0
    @inbounds for iX ∈ 1:solution_num_nz
        iRow = solution_index[iX]
        ftran_solution_error = max(ftran_solution_error,
            abs(ftran_rhs.array[iRow] - solution_value[iX]))
        btran_solution_error = max(btran_solution_error,
            abs(btran_rhs.array[iRow] - solution_value[iX]))
    end
    return max(ftran_solution_error, btran_solution_error)
end

"""
    reinvert_on_numerical_trouble!(e, alpha_col, alpha_row, tol)

`HEkk::reinvertOnNumericalTrouble`: compares pivots computed by column and
by row, increases Markowitz threshold if needed, and returns `(reinvert, measure)`.
Returns `false` if no updates have been performed yet.
"""
function reinvert_on_numerical_trouble!(e::SimplexEngine,
    alpha_from_col::Float64, alpha_from_row::Float64, tol::Float64)
    abs_col = abs(alpha_from_col)
    abs_row = abs(alpha_from_row)
    min_abs_alpha = min(abs_col, abs_row)
    abs_alpha_diff = abs(abs_col - abs_row)
    numerical_trouble_measure = abs_alpha_diff / min_abs_alpha
    reinvert = numerical_trouble_measure > tol && e.info.update_count > 0
    if reinvert
        current = e.info.factor_pivot_threshold
        new_threshold = 0.0
        if current < kDefaultPivotThreshold
            new_threshold = min(current * kPivotThresholdChangeFactor,
                kDefaultPivotThreshold)
        elseif current < kMaxPivotThreshold && e.info.update_count < 10
            new_threshold = min(current * kPivotThresholdChangeFactor,
                kMaxPivotThreshold)
        end
        if new_threshold != 0.0
            e.info.factor_pivot_threshold = new_threshold
            e.nla.factor.pivot_threshold = new_threshold
        end
    end
    return reinvert, numerical_trouble_measure
end

"""`HEkk::flipBound` — flips nonbasic variable bound."""
function flip_bound!(e::SimplexEngine, iCol::Int)
    move = -e.basis.nonbasicMove[iCol]
    e.basis.nonbasicMove[iCol] = move
    e.info.workValue[iCol] = move == 1 ? e.info.workLower[iCol] :
                             e.info.workUpper[iCol]
    return e
end

"""
    update_factor!(e, column, row_ep, iRow, hint)

`HEkk::updateFactor`: FT update of factor, then checks reinversion triggers
(update limit, synthetic clock). `hint` is current `rebuild_reason`;
returns updated hint.
"""
function update_factor!(e::SimplexEngine, column::HVector, row_ep::HVector,
    iRow::Int, hint::Int)
    update!(e.nla, column, row_ep, iRow)
    e.status.has_invert = true
    e.status.has_fresh_invert = false
    if e.info.update_count >= e.info.update_limit
        hint = kRebuildReasonUpdateLimitReached
    end
    reinvert_synthetic_clock = e.total_synthetic_tick >= e.build_synthetic_tick
    performed_min_updates =
        e.info.update_count >= kSyntheticTickReinversionMinUpdateCount
    if reinvert_synthetic_clock && performed_min_updates
        hint = kRebuildReasonSyntheticClockSaysInvert
    end
    return hint
end

"""`HEkk::updatePivots` — basis entry/exit, counters and flags."""
function update_pivots!(e::SimplexEngine, variable_in::Int, row_out::Int,
    move_out::Int)
    basis = e.basis
    info = e.info
    variable_out = basis.basicIndex[row_out]
    # Basis hash (cycling detection): outgoing then incoming.
    basis.hash = sparse_inverse_combine(basis.hash, variable_out - 1)
    basis.hash = sparse_combine(basis.hash, variable_in - 1)
    push!(e.visited_basis, basis.hash)
    basis.basicIndex[row_out] = variable_in
    basis.nonbasicFlag[variable_in] = kNonbasicFlagFalse
    basis.nonbasicMove[variable_in] = kNonbasicMoveZe
    info.baseLower[row_out] = info.workLower[variable_in]
    info.baseUpper[row_out] = info.workUpper[variable_in]
    basis.nonbasicFlag[variable_out] = kNonbasicFlagTrue
    if info.workLower[variable_out] == info.workUpper[variable_out]
        info.workValue[variable_out] = info.workLower[variable_out]
        basis.nonbasicMove[variable_out] = kNonbasicMoveZe
    elseif move_out == -1
        info.workValue[variable_out] = info.workLower[variable_out]
        basis.nonbasicMove[variable_out] = kNonbasicMoveUp
    else
        info.workValue[variable_out] = info.workUpper[variable_out]
        basis.nonbasicMove[variable_out] = kNonbasicMoveDn
    end
    info.updated_dual_objective_value +=
        info.workValue[variable_out] * info.workDual[variable_out]
    info.update_count += 1
    if variable_out < e.lp.num_col
        info.num_basic_logicals += 1
    end
    if variable_in < e.lp.num_col
        info.num_basic_logicals -= 1
    end
    e.status.has_invert = false
    e.status.has_fresh_invert = false
    e.status.has_fresh_rebuild = false
    return e
end

"""`HEkk::updateMatrix` — updates partitioned row-wise view."""
function update_matrix!(e::SimplexEngine, variable_in::Int, variable_out::Int)
    update!(e.ar_matrix, variable_in, variable_out, e.lp.a_matrix)
    return e
end

"""`HEkk::initialisePartitionedRowwiseMatrix` — row-wise view of nonbasics."""
function initialise_partitioned_rowwise_matrix!(e::SimplexEngine)
    e.status.has_ar_matrix && return e
    in_partition = [e.basis.nonbasicFlag[i] == kNonbasicFlagTrue
                    for i ∈ 1:(e.lp.num_col + e.lp.num_row)]
    create_rowwise_partitioned!(e.ar_matrix, e.lp.a_matrix, in_partition)
    e.status.has_ar_matrix = true
    return e
end

"""`HEkk::computeSimplexPrimalInfeasible` — count/max/sum of primal infeasibilities."""
function compute_simplex_primal_infeasible!(e::SimplexEngine)
    info = e.info
    basis = e.basis
    tolerance = e.options.primal_feasibility_tolerance
    info.num_primal_infeasibilities = 0
    info.max_primal_infeasibility = 0.0
    info.sum_primal_infeasibilities = 0.0
    for i ∈ 1:(e.lp.num_col + e.lp.num_row)
        basis.nonbasicFlag[i] == kNonbasicFlagTrue || continue
        value = info.workValue[i]
        lower = info.workLower[i]
        upper = info.workUpper[i]
        primal_infeasibility = 0.0
        if value < lower - tolerance
            primal_infeasibility = lower - value
        elseif value > upper + tolerance
            primal_infeasibility = value - upper
        end
        if primal_infeasibility > 0
            if primal_infeasibility > tolerance
                info.num_primal_infeasibilities += 1
            end
            info.max_primal_infeasibility =
                max(primal_infeasibility, info.max_primal_infeasibility)
            info.sum_primal_infeasibilities += primal_infeasibility
        end
    end
    for i ∈ 1:e.lp.num_row
        value = info.baseValue[i]
        lower = info.baseLower[i]
        upper = info.baseUpper[i]
        primal_infeasibility = 0.0
        if value < lower - tolerance
            primal_infeasibility = lower - value
        elseif value > upper + tolerance
            primal_infeasibility = value - upper
        end
        if primal_infeasibility > 0
            if primal_infeasibility > tolerance
                info.num_primal_infeasibilities += 1
            end
            info.max_primal_infeasibility =
                max(primal_infeasibility, info.max_primal_infeasibility)
            info.sum_primal_infeasibilities += primal_infeasibility
        end
    end
    return e
end

"""`HEkk::computeSimplexDualInfeasible` — dual infeasibilities based on `nonbasicMove`."""
function compute_simplex_dual_infeasible!(e::SimplexEngine)
    info = e.info
    basis = e.basis
    tolerance = e.options.dual_feasibility_tolerance
    info.num_dual_infeasibilities = 0
    info.max_dual_infeasibility = 0.0
    info.sum_dual_infeasibilities = 0.0
    for iCol ∈ 1:(e.lp.num_col + e.lp.num_row)
        basis.nonbasicFlag[iCol] == kNonbasicFlagTrue || continue
        dual = info.workDual[iCol]
        lower = info.workLower[iCol]
        upper = info.workUpper[iCol]
        if lower == -kHighsInf && upper == kHighsInf
            dual_infeasibility = abs(dual)
        else
            dual_infeasibility = -basis.nonbasicMove[iCol] * dual
        end
        if dual_infeasibility > 0
            if dual_infeasibility >= tolerance
                info.num_dual_infeasibilities += 1
            end
            info.max_dual_infeasibility =
                max(dual_infeasibility, info.max_dual_infeasibility)
            info.sum_dual_infeasibilities += dual_infeasibility
        end
    end
    return e
end

"""`HEkk::computeSimplexInfeasible`."""
function compute_simplex_infeasible!(e::SimplexEngine)
    compute_simplex_primal_infeasible!(e)
    compute_simplex_dual_infeasible!(e)
    return e
end

"""
    return_from_solve!(e, algorithm)

`HEkk::returnFromSolve`: normalized state returned by a solve. Removes shifts
and perturbations, recomputes primal values, duals, and infeasibilities based
on status, resets `valid_backtracking_basis` to false, and zeroes basic duals.
"""
function return_from_solve!(e::SimplexEngine, algorithm::SimplexAlgorithm)
    status = e.model_status
    info = e.info
    info.valid_backtracking_basis = false
    status == kSolveError && return e
    if status != kOptimal
        invalidate_primal_infeasibility_record!(e)
        invalidate_dual_infeasibility_record!(e)
    end
    if status == kInfeasible
        if algorithm == kPrimal
            # After proven infeasibility in primal phase 1, duals are
            # recomputed using LP costs.
            initialise_cost!(e, kDual, kSolvePhase2)
            compute_dual!(e)
        end
        compute_simplex_infeasible!(e)
    elseif status == kUnboundedOrInfeasible
        # LP bounds, primals, and infeasibilities recomputed: primal will decide.
        initialise_bound!(e, kDual, kSolvePhase2)
        compute_primal!(e)
        compute_simplex_infeasible!(e)
    elseif status == kUnbounded
        compute_simplex_infeasible!(e)
    elseif status != kOptimal
        # Limit reached (iterations, time, objective) or unknown status:
        # LP bounds and costs, values and duals recomputed.
        initialise_bound!(e, kDual, kSolvePhase2)
        initialise_nonbasic_value_and_move!(e)
        compute_primal!(e)
        initialise_cost!(e, kDual, kSolvePhase2)
        compute_dual!(e)
        compute_simplex_infeasible!(e)
    end
    for iRow ∈ 1:e.lp.num_row
        info.workDual[e.basis.basicIndex[iRow]] = 0.0
    end
    compute_primal_objective_value!(e)
    # `HApp::solveLpSimplex`: returned basis is cached for warm start.
    store_solution_basis!(e)
    return e
end

"""
`HEkk::computeSimplexLpDualInfeasible` — dual infeasibilities with respect to LP bounds
(used to conclude in phase 1). Returns `(num, max, sum)`.
"""
function compute_simplex_lp_dual_infeasible(e::SimplexEngine)
    info = e.info
    basis = e.basis
    tolerance = e.options.dual_feasibility_tolerance
    num = 0
    max_infeasibility = 0.0
    sum_infeasibility = 0.0
    for iCol ∈ 1:e.lp.num_col
        basis.nonbasicFlag[iCol] == kNonbasicFlagTrue || continue
        dual = info.workDual[iCol]
        lower = e.lp.col_lower[iCol]
        upper = e.lp.col_upper[iCol]
        if upper == kHighsInf
            dual_infeasibility = lower == -kHighsInf ? abs(dual) : -dual
        else
            dual_infeasibility = lower == -kHighsInf ? dual : 0.0
        end
        if dual_infeasibility > 0
            if dual_infeasibility >= tolerance
                num += 1
            end
            max_infeasibility = max(dual_infeasibility, max_infeasibility)
            sum_infeasibility += dual_infeasibility
        end
    end
    for iRow ∈ 1:e.lp.num_row
        iVar = e.lp.num_col + iRow
        basis.nonbasicFlag[iVar] == kNonbasicFlagTrue || continue
        dual = -info.workDual[iVar]
        lower = e.lp.row_lower[iRow]
        upper = e.lp.row_upper[iRow]
        if upper == kHighsInf
            dual_infeasibility = lower == -kHighsInf ? abs(dual) : -dual
        else
            dual_infeasibility = lower == -kHighsInf ? dual : 0.0
        end
        if dual_infeasibility > 0
            if dual_infeasibility >= tolerance
                num += 1
            end
            max_infeasibility = max(dual_infeasibility, max_infeasibility)
            sum_infeasibility += dual_infeasibility
        end
    end
    return num, max_infeasibility, sum_infeasibility
end

"""`HEkk::invalidatePrimalInfeasibilityRecord`."""
function invalidate_primal_infeasibility_record!(e::SimplexEngine)
    e.info.num_primal_infeasibilities = kHighsIllegalInfeasibilityCount
    e.info.max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure
    e.info.sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure
    return e
end

"""`HEkk::invalidatePrimalMaxSumInfeasibilityRecord`."""
function invalidate_primal_max_sum_infeasibility_record!(e::SimplexEngine)
    e.info.max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure
    e.info.sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure
    return e
end

"""
    apply_taboo_variable_in!(e, values, overwrite_with)

`HEkk::applyTabooVariableIn`: masks values of taboo incoming variables
(making them unattractive during CHUZC). `values` is typically `workDual`.
"""
function apply_taboo_variable_in!(e::SimplexEngine, values::Vector{Float64},
    overwrite_with::Float64)
    for change ∈ e.bad_basis_change
        if change.taboo
            change.save_value = values[change.variable_in]
            values[change.variable_in] = overwrite_with
        end
    end
    return e
end

"""`HEkk::unapplyTabooVariableIn` — traverses in reverse order (matching source)."""
function unapply_taboo_variable_in!(e::SimplexEngine,
    values::Vector{Float64})
    for iX ∈ length(e.bad_basis_change):-1:1
        change = e.bad_basis_change[iX]
        if change.taboo
            values[change.variable_in] = change.save_value
        end
    end
    return e
end

"""`HEkk::invalidateDualInfeasibilityRecord`."""
function invalidate_dual_infeasibility_record!(e::SimplexEngine)
    e.info.num_dual_infeasibilities = kHighsIllegalInfeasibilityCount
    e.info.max_dual_infeasibility = kHighsIllegalInfeasibilityMeasure
    e.info.sum_dual_infeasibilities = kHighsIllegalInfeasibilityMeasure
    return e
end

"""
    bailout!(e)

`HEkk::bailout`: checks iteration limit and time limit. Sets `solve_bailout`
and model status when a limit is reached.
"""
function bailout!(e::SimplexEngine)
    e.solve_bailout && return true
    if e.options.time_limit < kHighsInf &&
       (time() - e.solve_start_time) > e.options.time_limit
        e.solve_bailout = true
        e.model_status = kTimeLimit
    elseif e.iteration_count - e.iteration_count0 >=
           e.options.simplex_iteration_limit
        e.solve_bailout = true
        e.model_status = kIterationLimit
    end
    return e.solve_bailout
end

"""`HEkk::choosePriceTechnique` — column or row-wise price with switch."""
function choose_price_technique(e::SimplexEngine, row_ep_density::Float64)
    density_for_column_price_switch = 0.75
    price_strategy = e.info.price_strategy
    use_col_price = price_strategy == kSimplexPriceStrategyCol ||
                    (price_strategy == kSimplexPriceStrategyRowSwitchColSwitch &&
                     row_ep_density > density_for_column_price_switch)
    use_row_price_w_switch =
        price_strategy == kSimplexPriceStrategyRowSwitchColSwitch
    return use_col_price, use_row_price_w_switch
end

"""
    tableau_row_price!(e, row_ep, row_ap, quad_precision = false)

`HEkk::tableauRowPrice`: `row_ap = row_ep' A` over nonbasic variables, via
`priceByColumn` or `priceByRowWithSwitch` (partitioned view), followed by
exponential moving average of `row_ap` density. `quad_precision` accumulates
in double-double (compensated quad precision) and requires `row_ep` to be in factor scale.
"""
function tableau_row_price!(e::SimplexEngine, row_ep::HVector, row_ap::HVector,
    quad_precision::Bool=false)
    solver_num_row = e.lp.num_row
    local_density = row_ep.count / solver_num_row
    use_col_price, use_row_price_w_switch = choose_price_technique(e,
        local_density)
    clear!(row_ap)
    if use_col_price
        price_by_column!(e.lp.a_matrix, row_ap, row_ep, quad_precision)
        for iCol ∈ 1:e.lp.num_col
            row_ap.array[iCol] *= e.basis.nonbasicFlag[iCol]
        end
    elseif use_row_price_w_switch
        price_by_row_with_switch!(e.ar_matrix, row_ap, row_ep,
            e.info.row_ap_density, 1, kHyperPriceDensity, quad_precision)
    else
        price_by_row!(e.ar_matrix, row_ap, row_ep, quad_precision)
    end
    e.info.row_ap_density = update_operation_result_density(
        e.info.row_ap_density, row_ap.count / e.lp.num_col)
    return row_ap
end

"""
    unit_btran_residual!(e, row_out, row_ep, residual)

`HEkk::unitBtranResidual`: `residual = e_row_out - Bᵀ row_ep`, accumulated in
double-double (residual is a difference of close quantities; in Float64 rounding
noise dominates). Returns infinity norm of residual.
"""
function unit_btran_residual!(e::SimplexEngine, row_out::Int, row_ep::HVector,
    residual::HVector)
    lp = e.lp
    quad_residual = zeros(CDouble, lp.num_row)
    quad_residual[row_out] = CDouble(-1.0)
    for iRow ∈ 1:lp.num_row
        iVar = e.basis.basicIndex[iRow]
        value = quad_residual[iRow]
        if iVar <= lp.num_col
            for iEl ∈ lp.a_matrix.start[iVar]:(lp.a_matrix.start[iVar + 1] - 1)
                value += lp.a_matrix.value[iEl] *
                         row_ep.array[lp.a_matrix.index[iEl]]
            end
        else
            value += row_ep.array[iVar - lp.num_col]
        end
        quad_residual[iRow] = value
    end
    clear!(residual)
    residual.packFlag = false
    residual_norm = 0.0
    for iRow ∈ 1:lp.num_row
        value = Float64(quad_residual[iRow])
        if value != 0.0
            residual.array[iRow] = value
            residual.count += 1
            residual.index[residual.count] = iRow
        end
        residual_norm = max(abs(residual.array[iRow]), residual_norm)
    end
    return residual_norm
end

"""
    unit_btran_iterative_refinement!(e, row_out, row_ep)

`HEkk::unitBtranIterativeRefinement`: one pass of iterative refinement on unit
BTRAN `row_ep`. Residual is scaled by nearest power of two before BTRAN (so `kHighsTiny`
does not zero out small residuals), then correction is subtracted and index list
is reconstructed in ascending row order.
"""
function unit_btran_iterative_refinement!(e::SimplexEngine, row_out::Int,
    row_ep::HVector)
    residual = HVector(e.lp.num_row)
    residual_norm = unit_btran_residual!(e, row_out, row_ep, residual)
    residual_norm == 0.0 && return e
    residual_scale = nearest_power_of_two_scale(residual_norm)
    for iEl ∈ 1:residual.count
        residual.array[residual.index[iEl]] *= residual_scale
    end
    btran!(e.nla, residual, 1.0)
    row_ep.count = 0
    for iRow ∈ 1:e.lp.num_row
        if residual.array[iRow] != 0.0
            correction_value = residual.array[iRow] / residual_scale
            row_ep.array[iRow] -= correction_value
        end
        if abs(row_ep.array[iRow]) < kHighsTiny
            row_ep.array[iRow] = 0.0
        else
            row_ep.count += 1
            row_ep.index[row_ep.count] = iRow
        end
    end
    return e
end

"""`nearestPowerOfTwoScale` (`HighsUtils.cpp:1211`)."""
function nearest_power_of_two_scale(value::Float64)
    check_x, exp_scale = frexp(value)
    if abs(check_x) == 0.5
        check_x *= 2
        exp_scale -= 1
    end
    return ldexp(1.0, -exp_scale)
end

"""`HEkk::getValueScale` — pivot scale (nearest power of two)."""
function get_value_scale(count::Int, value::AbstractVector{Float64})
    count <= 0 && return 1.0
    max_abs_value = 0.0
    for iX ∈ 1:count
        max_abs_value = max(abs(value[iX]), max_abs_value)
    end
    return nearest_power_of_two_scale(max_abs_value)
end

"""`HEkk::getMaxAbsRowValue` (row-wise view initialized if needed)."""
function get_max_abs_row_value(e::SimplexEngine, row::Int)
    initialise_partitioned_rowwise_matrix!(e)
    val = -1.0
    for i ∈ e.ar_matrix.start[row]:(e.ar_matrix.start[row + 1] - 1)
        val = max(val, abs(e.ar_matrix.value[i]))
    end
    return val
end

"""
    proof_of_primal_infeasibility!(e, row_ep, move_out, row_out)

`HEkk::proofOfPrimalInfeasibility`: finds a linear combination `y = row_ep' A`
whose implied upper bound contradicts the constraint lower bound. Accumulators
`proof_lower`/`implied_upper`/`sumInf` use `CDouble` (compensated double-double,
as in C++ source) to prevent rounding noise from creating false infeasibility proofs.
"""
function proof_of_primal_infeasibility!(e::SimplexEngine, row_ep::HVector,
    move_out::Int, row_out::Int)
    lp = e.lp
    proof_lower = CDouble(0.0)
    # Refine row_ep: purge negligible contributions and infinite bounds
    for iX ∈ 1:row_ep.count
        iRow = row_ep.index[iX]
        row_ep_value = row_ep.array[iRow]
        row_ep_value == 0.0 && continue
        if abs(row_ep_value * get_max_abs_row_value(e, iRow)) <=
           e.options.small_matrix_value
            row_ep.array[iRow] = 0.0
            continue
        end
        row_ep.array[iRow] *= move_out
        if row_ep.array[iRow] > 0
            rowBound = lp.row_lower[iRow]
            if rowBound == -kHighsInf
                row_ep.array[iRow] = 0.0
                continue
            end
        else
            rowBound = lp.row_upper[iRow]
            if rowBound == kHighsInf
                row_ep.array[iRow] = 0.0
                continue
            end
        end
        proof_lower += row_ep.array[iRow] * rowBound
    end
    # Proof coefficients: row_ep' ar_matrix, accumulated in double.
    proof_scattered = zeros(lp.num_col)
    for iRow ∈ 1:lp.num_row
        multiplier = row_ep.array[iRow]
        multiplier == 0.0 && continue
        for iEl ∈ e.ar_matrix.start[iRow]:(e.ar_matrix.start[iRow + 1] - 1)
            iCol = e.ar_matrix.index[iEl]
            proof_scattered[iCol] += multiplier * e.ar_matrix.value[iEl]
        end
    end
    proof_value = Float64[]
    proof_index = Int[]
    for iCol ∈ 1:lp.num_col
        value = proof_scattered[iCol]
        if abs(value) > kHighsTiny
            push!(proof_value, value)
            push!(proof_index, iCol)
        end
    end
    implied_upper = CDouble(0.0)
    sum_inf = CDouble(0.0)
    for i ∈ eachindex(proof_value)
        iCol = proof_index[i]
        value = proof_value[i]
        if value > 0
            if lp.col_upper[iCol] == kHighsInf
                sum_inf += value
                sum_inf > e.options.small_matrix_value && break
                continue
            end
            implied_upper += value * lp.col_upper[iCol]
        else
            if lp.col_lower[iCol] == -kHighsInf
                sum_inf += -value
                sum_inf > e.options.small_matrix_value && break
                continue
            end
            implied_upper += value * lp.col_lower[iCol]
        end
    end
    infinite_implied_upper = sum_inf > e.options.small_matrix_value
    gap = Float64(proof_lower - implied_upper)
    gap_ok = gap > e.options.primal_feasibility_tolerance
    return !infinite_implied_upper && gap_ok
end
