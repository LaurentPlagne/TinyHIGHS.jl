# Port of `lp_data/HighsLpUtils.{h,cpp}` (LP scaling) — M5b: decision
# (`considerScaling`), computation (`scaleLp`, `equilibrationScaleMatrix`,
# `maxValueScaleMatrix`), and applying/unapplying factors (`HighsLp::applyScale`/
# `unapplyScale`/`clearScale`). The implementation replicates exact abandonment
# criteria (`improvement_factor`), so identical matrices are scaled or unscaled.
#
# Scale factors are powers of two: `apply` then `unapply` restores the
# matrix bit-for-bit. The model returned to the caller remains unscaled, matching
# `Highs::run`; `unscaleSimplex` (`HEkk`) converts working space back to original
# units on exiting the solve.
#
# Not ported: user cost scales (`user_objective_scale`/`user_bound_scale`),
# `scaleSimplexCost`, reporting/analysis.

"""
`HighsLp::clearScale`: clears scale factors (Off strategy, unscaled).
"""
function clear_scale!(lp::SimplexLp)
    empty!(lp.scale.col)
    empty!(lp.scale.row)
    lp.scale = Scale()
    return lp
end
"""
`HighsLp::applyScale`: applies scale factors to model (`is_scaled`) if
present; no-op if already applied or absent.
"""
function apply_lp_scale!(lp::SimplexLp)
    lp.is_scaled && return lp
    scale = lp.scale
    scale.has_scaling || return lp
    for iCol ∈ 1:lp.num_col
        lp.col_lower[iCol] /= scale.col[iCol]
        lp.col_upper[iCol] /= scale.col[iCol]
        lp.col_cost[iCol] *= scale.col[iCol]
    end
    for iRow ∈ 1:lp.num_row
        lp.row_lower[iRow] *= scale.row[iRow]
        lp.row_upper[iRow] *= scale.row[iRow]
    end
    apply_scale!(lp.a_matrix, scale)
    lp.is_scaled = true
    return lp
end

"""
`HighsLp::unapplyScale`: removes scale factors from model (inverse operation
is exact: powers of two).
"""
function unapply_lp_scale!(lp::SimplexLp)
    lp.is_scaled || return lp
    scale = lp.scale
    for iCol ∈ 1:lp.num_col
        lp.col_lower[iCol] *= scale.col[iCol]
        lp.col_upper[iCol] *= scale.col[iCol]
        lp.col_cost[iCol] /= scale.col[iCol]
    end
    for iRow ∈ 1:lp.num_row
        lp.row_lower[iRow] *= scale.row[iRow]
        lp.row_upper[iRow] /= scale.row[iRow]
    end
    unapply_scale!(lp.a_matrix, scale)
    lp.is_scaled = false
    return lp
end

"""`HighsLp::clearScaling` — remove and clear scale factors."""
function clear_scaling!(lp::SimplexLp)
    unapply_lp_scale!(lp)
    clear_scale!(lp)
    return lp
end

"""
    equilibration_scale_matrix!(lp, strategy, allowed_matrix_scale_factor) -> Bool

`equilibrationScaleMatrix`: six equilibration passes across columns/rows using
`1/sqrt(min·max)`, factors rounded to nearest power of two and bounded by
`2^±allowed_matrix_scale_factor`, applied to matrix followed by abandonment check
(except `ForcedEquilibration`): product of average, extreme, and max/min ratio
improvements must be >= 1.
"""
function equilibration_scale_matrix!(lp::SimplexLp, strategy::Int,
    allowed_matrix_scale_factor::Int)
    num_col, num_row = lp.num_col, lp.num_row
    scale = lp.scale
    col_scale, row_scale = scale.col, scale.row
    m = lp.a_matrix
    col_cost = lp.col_cost

    # Rather than retaining full statistics from source (for reporting),
    # only the aggregates that govern abandonment decisions are computed.
    original_matrix_min_value = kHighsInf
    original_matrix_max_value = 0.0
    for k ∈ 1:num_nz(m)
        value = abs(m.value[k])
        original_matrix_min_value = min(original_matrix_min_value, value)
        original_matrix_max_value = max(original_matrix_max_value, value)
    end

    min_nonzero_cost = kHighsInf
    for iCol ∈ 1:num_col
        if col_cost[iCol] != 0.0
            min_nonzero_cost = min(abs(col_cost[iCol]), min_nonzero_cost)
        end
    end
    include_cost_in_scaling = min_nonzero_cost < 0.1

    max_allow_scale = 2.0^allowed_matrix_scale_factor
    min_allow_scale = 1 / max_allow_scale
    min_allow_col_scale = min_allow_scale
    max_allow_col_scale = max_allow_scale
    min_allow_row_scale = min_allow_scale
    max_allow_row_scale = max_allow_scale

    finite_infinity = 1e200
    row_min_value = fill(finite_infinity, num_row)
    row_max_value = fill(1 / finite_infinity, num_row)
    for _ ∈ 1:6
        # Column scaling, collect row bounds.
        for iCol ∈ 1:num_col
            col_min_value = finite_infinity
            col_max_value = 1 / finite_infinity
            abs_col_cost = abs(col_cost[iCol])
            if include_cost_in_scaling && abs_col_cost != 0.0
                col_min_value = min(col_min_value, abs_col_cost)
                col_max_value = max(col_max_value, abs_col_cost)
            end
            for k ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                value = abs(m.value[k]) * row_scale[m.index[k]]
                col_min_value = min(col_min_value, value)
                col_max_value = max(col_max_value, value)
            end
            col_equilibration = 1 / sqrt(col_min_value * col_max_value)
            col_scale[iCol] = min(max(min_allow_col_scale, col_equilibration),
                max_allow_col_scale)
            for k ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                iRow = m.index[k]
                value = abs(m.value[k]) * col_scale[iCol]
                row_min_value[iRow] = min(row_min_value[iRow], value)
                row_max_value[iRow] = max(row_max_value[iRow], value)
            end
        end
        # Row scaling.
        for iRow ∈ 1:num_row
            row_equilibration =
                1 / sqrt(row_min_value[iRow] * row_max_value[iRow])
            row_scale[iRow] = min(max(min_allow_row_scale, row_equilibration),
                max_allow_row_scale)
        end
        fill!(row_min_value, finite_infinity)
        fill!(row_max_value, 1 / finite_infinity)
    end
    # Nearest power of two.
    log2 = log(2.0)
    for iCol ∈ 1:num_col
        col_scale[iCol] = 2.0^floor(log(col_scale[iCol]) / log2 + 0.5)
    end
    for iRow ∈ 1:num_row
        row_scale[iRow] = 2.0^floor(log(row_scale[iRow]) / log2 + 0.5)
    end
    # Apply to matrix, with before/after equilibration statistics.
    matrix_min_value = finite_infinity
    matrix_max_value = 0.0
    min_original_col_equilibration = finite_infinity
    sum_original_log_col_equilibration = 0.0
    max_original_col_equilibration = 0.0
    min_original_row_equilibration = finite_infinity
    sum_original_log_row_equilibration = 0.0
    max_original_row_equilibration = 0.0
    min_col_equilibration = finite_infinity
    sum_log_col_equilibration = 0.0
    max_col_equilibration = 0.0
    min_row_equilibration = finite_infinity
    sum_log_row_equilibration = 0.0
    max_row_equilibration = 0.0
    original_row_min_value = fill(finite_infinity, num_row)
    original_row_max_value = fill(1 / finite_infinity, num_row)
    row_min_value = fill(finite_infinity, num_row)
    row_max_value = fill(1 / finite_infinity, num_row)
    for iCol ∈ 1:num_col
        original_col_min_value = finite_infinity
        original_col_max_value = 1 / finite_infinity
        col_min_value = finite_infinity
        col_max_value = 1 / finite_infinity
        for k ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
            iRow = m.index[k]
            original_value = abs(m.value[k])
            original_col_min_value = min(original_value, original_col_min_value)
            original_col_max_value = max(original_value, original_col_max_value)
            original_row_min_value[iRow] =
                min(original_row_min_value[iRow], original_value)
            original_row_max_value[iRow] =
                max(original_row_max_value[iRow], original_value)
            m.value[k] *= col_scale[iCol] * row_scale[iRow]
            value = abs(m.value[k])
            col_min_value = min(value, col_min_value)
            col_max_value = max(value, col_max_value)
            row_min_value[iRow] = min(row_min_value[iRow], value)
            row_max_value[iRow] = max(row_max_value[iRow], value)
        end
        matrix_min_value = min(matrix_min_value, col_min_value)
        matrix_max_value = max(matrix_max_value, col_max_value)
        original_col_equilibration =
            1 / sqrt(original_col_min_value * original_col_max_value)
        min_original_col_equilibration =
            min(original_col_equilibration, min_original_col_equilibration)
        sum_original_log_col_equilibration += log(original_col_equilibration)
        max_original_col_equilibration =
            max(original_col_equilibration, max_original_col_equilibration)
        col_equilibration = 1 / sqrt(col_min_value * col_max_value)
        min_col_equilibration = min(col_equilibration, min_col_equilibration)
        sum_log_col_equilibration += log(col_equilibration)
        max_col_equilibration = max(col_equilibration, max_col_equilibration)
    end
    for iRow ∈ 1:num_row
        original_row_equilibration =
            1 / sqrt(original_row_min_value[iRow] * original_row_max_value[iRow])
        min_original_row_equilibration =
            min(original_row_equilibration, min_original_row_equilibration)
        sum_original_log_row_equilibration += log(original_row_equilibration)
        max_original_row_equilibration =
            max(original_row_equilibration, max_original_row_equilibration)
        row_equilibration = 1 / sqrt(row_min_value[iRow] * row_max_value[iRow])
        min_row_equilibration = min(row_equilibration, min_row_equilibration)
        sum_log_row_equilibration += log(row_equilibration)
        max_row_equilibration = max(row_equilibration, max_row_equilibration)
    end
    geomean_original_col_equilibration =
        exp(sum_original_log_col_equilibration / num_col)
    geomean_original_row_equilibration =
        exp(sum_original_log_row_equilibration / num_row)
    geomean_col_equilibration = exp(sum_log_col_equilibration / num_col)
    geomean_row_equilibration = exp(sum_log_row_equilibration / num_row)
    geomean_original_col =
        max(geomean_original_col_equilibration,
            1 / geomean_original_col_equilibration)
    geomean_original_row =
        max(geomean_original_row_equilibration,
            1 / geomean_original_row_equilibration)
    geomean_col = max(geomean_col_equilibration, 1 / geomean_col_equilibration)
    geomean_row = max(geomean_row_equilibration, 1 / geomean_row_equilibration)
    mean_equilibration_improvement =
        sqrt((geomean_original_col * geomean_original_row) /
             (geomean_col * geomean_row))
    original_col_ratio =
        max_original_col_equilibration / min_original_col_equilibration
    original_row_ratio =
        max_original_row_equilibration / min_original_row_equilibration
    col_ratio = max_col_equilibration / min_col_equilibration
    row_ratio = max_row_equilibration / min_row_equilibration
    extreme_equilibration_improvement =
        (original_col_ratio + original_row_ratio) / (col_ratio + row_ratio)
    matrix_value_ratio = matrix_max_value / matrix_min_value
    original_matrix_value_ratio =
        original_matrix_max_value / original_matrix_min_value
    matrix_value_ratio_improvement =
        original_matrix_value_ratio / matrix_value_ratio
    possibly_abandon_scaling =
        strategy != kSimplexScaleStrategyForcedEquilibration
    improvement_factor = extreme_equilibration_improvement *
                         mean_equilibration_improvement *
                         matrix_value_ratio_improvement
    poor_improvement = improvement_factor < 1.0
    if possibly_abandon_scaling && poor_improvement
        for iCol ∈ 1:num_col
            for k ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                m.value[k] /= col_scale[iCol] * row_scale[m.index[k]]
            end
        end
        return false
    end
    return true
end

"""
    max_value_scale_matrix!(lp, allowed_matrix_scale_factor) -> Bool

`maxValueScaleMatrix` (strategy 4): row scaling by `2^round(log2(1/max))`,
then column scaling similarly, apply scaling, and abort if the max/min ratio
of values does not improve (`<= 1`).
"""
function max_value_scale_matrix!(lp::SimplexLp,
    allowed_matrix_scale_factor::Int)
    num_col, num_row = lp.num_col, lp.num_row
    scale = lp.scale
    col_scale, row_scale = scale.col, scale.row
    m = lp.a_matrix
    log2 = log(2.0)
    max_allow_scale = 2.0^allowed_matrix_scale_factor
    min_allow_scale = 1 / max_allow_scale
    min_allow_col_scale = min_allow_scale
    max_allow_col_scale = max_allow_scale
    min_allow_row_scale = min_allow_scale
    max_allow_row_scale = max_allow_scale

    original_matrix_min_value = kHighsInf
    original_matrix_max_value = 0.0
    row_max_value = zeros(num_row)
    for iCol ∈ 1:num_col
        for k ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
            iRow = m.index[k]
            value = abs(m.value[k])
            row_max_value[iRow] = max(row_max_value[iRow], value)
            original_matrix_min_value = min(original_matrix_min_value, value)
            original_matrix_max_value = max(original_matrix_max_value, value)
        end
    end
    for iRow ∈ 1:num_row
        if row_max_value[iRow] != 0.0
            row_scale_value = 1 / row_max_value[iRow]
            row_scale_value = 2.0^floor(log(row_scale_value) / log2 + 0.5)
            row_scale_value = min(max(min_allow_row_scale, row_scale_value),
                max_allow_row_scale)
            row_scale[iRow] = row_scale_value
        end
    end
    matrix_min_value = kHighsInf
    matrix_max_value = 0.0
    for iCol ∈ 1:num_col
        col_max_value = 0.0
        for k ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
            iRow = m.index[k]
            m.value[k] *= row_scale[iRow]
            value = abs(m.value[k])
            col_max_value = max(col_max_value, value)
        end
        if col_max_value != 0.0
            col_scale_value = 1 / col_max_value
            col_scale_value = 2.0^floor(log(col_scale_value) / log2 + 0.5)
            col_scale_value = min(max(min_allow_col_scale, col_scale_value),
                max_allow_col_scale)
            col_scale[iCol] = col_scale_value
            for k ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                m.value[k] *= col_scale[iCol]
                value = abs(m.value[k])
                matrix_min_value = min(matrix_min_value, value)
                matrix_max_value = max(matrix_max_value, value)
            end
        end
    end
    matrix_value_ratio = matrix_max_value / matrix_min_value
    original_matrix_value_ratio =
        original_matrix_max_value / original_matrix_min_value
    matrix_value_ratio_improvement =
        original_matrix_value_ratio / matrix_value_ratio
    improvement_factor = matrix_value_ratio_improvement
    poor_improvement = improvement_factor <= 1.0
    if poor_improvement
        for iCol ∈ 1:num_col
            for k ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                m.value[k] /= col_scale[iCol] * row_scale[m.index[k]]
            end
        end
        return false
    end
    return true
end

"""
    scale_lp!(lp; strategy, allowed_matrix_scale_factor, force_scaling=false)
        -> Bool

`scaleLp`: determines scale factors (`equilibration` or `maxValue`) then scales
bounds, costs and matrix if the matrix is scaled; `strategy == Choose` defaults to
`ForcedEquilibration`. The range `[0.2, 5]` short-circuits scaling.
The bound `allowed_matrix_scale_factor` is passed from options (the LP struct
does not carry options).
"""
function scale_lp!(lp::SimplexLp; strategy::Int,
    allowed_matrix_scale_factor::Int, force_scaling::Bool=false)
    clear_scaling!(lp)
    lp.num_col > 0 || return false
    use_scale_strategy =
        strategy == kSimplexScaleStrategyChoose ?
        kSimplexScaleStrategyForcedEquilibration : strategy
    original_matrix_min_value, original_matrix_max_value = range_abs(lp.a_matrix)
    no_scaling = force_scaling ? false :
                 (original_matrix_min_value >= 0.2) &&
                 (original_matrix_max_value <= 5.0)
    if !no_scaling
        lp.scale = Scale(ones(lp.num_col), ones(lp.num_row), 1.0, false,
            lp.num_col, lp.num_row, use_scale_strategy)
        equilibration_scaling =
            use_scale_strategy == kSimplexScaleStrategyEquilibration ||
            use_scale_strategy == kSimplexScaleStrategyForcedEquilibration
        scaled_matrix = equilibration_scaling ?
                        equilibration_scale_matrix!(lp, use_scale_strategy,
                            allowed_matrix_scale_factor) :
                        max_value_scale_matrix!(lp,
                            allowed_matrix_scale_factor)
        if scaled_matrix
            scale = lp.scale
            for iCol ∈ 1:lp.num_col
                lp.col_lower[iCol] /= scale.col[iCol]
                lp.col_upper[iCol] /= scale.col[iCol]
                lp.col_cost[iCol] *= scale.col[iCol]
            end
            for iRow ∈ 1:lp.num_row
                lp.row_lower[iRow] *= scale.row[iRow]
                lp.row_upper[iRow] *= scale.row[iRow]
            end
            lp.scale = Scale(scale.col, scale.row, 1.0, true, lp.num_col,
                lp.num_row, use_scale_strategy)
            lp.is_scaled = true
        else
            clear_scaling!(lp)
        end
    end
    # The source records the attempted strategy even when scaling was not applied.
    lp.scale = Scale(lp.scale.col, lp.scale.row, lp.scale.cost,
        lp.scale.has_scaling, lp.scale.num_col, lp.scale.num_row,
        use_scale_strategy)
    return lp.is_scaled
end

"""
    consider_scaling!(e) -> Bool

`considerScaling`: computes new scale factors if strategy changed (or has never
been attempted), otherwise reapplies known scale factors; removes scaling if no
longer allowed. Returns `true` if new factors were computed.
"""
function consider_scaling!(e::SimplexEngine)
    lp = e.lp
    options_strategy = e.options.simplex_scale_strategy
    new_scaling = false
    allow_scaling =
        lp.num_col > 0 && options_strategy != kSimplexScaleStrategyOff
    if lp.scale.has_scaling && !allow_scaling
        clear_scale!(lp)
        return true
    end
    scaling_not_tried = lp.scale.strategy == kSimplexScaleStrategyOff
    new_scaling_strategy =
        options_strategy != lp.scale.strategy &&
        options_strategy != kSimplexScaleStrategyChoose
    try_scaling = allow_scaling && (scaling_not_tried || new_scaling_strategy)
    if try_scaling
        unapply_lp_scale!(lp)
        scale_lp!(lp; strategy=options_strategy,
            allowed_matrix_scale_factor=e.options.allowed_matrix_scale_factor)
        new_scaling = lp.is_scaled
    elseif lp.scale.has_scaling
        apply_lp_scale!(lp)
    end
    return new_scaling
end

"""
    unscale_simplex!(e)

`HEkk::unscaleSimplex`: scales working space and base arrays back to original
units (the inverse of scales applied to the LP).
"""
function unscale_simplex!(e::SimplexEngine)
    lp = e.lp
    lp.is_scaled || return e
    num_col, num_row = lp.num_col, lp.num_row
    col_scale = lp.scale.col
    row_scale = lp.scale.row
    info = e.info
    for iCol ∈ 1:num_col
        factor = col_scale[iCol]
        info.workCost[iCol] /= factor
        info.workDual[iCol] /= factor
        info.workShift[iCol] /= factor
        info.workLower[iCol] *= factor
        info.workUpper[iCol] *= factor
        info.workRange[iCol] *= factor
        info.workValue[iCol] *= factor
        info.workLowerShift[iCol] *= factor
        info.workUpperShift[iCol] *= factor
    end
    for iRow ∈ 1:num_row
        iVar = num_col + iRow
        factor = row_scale[iRow]
        info.workCost[iVar] *= factor
        info.workDual[iVar] *= factor
        info.workShift[iVar] *= factor
        info.workLower[iVar] /= factor
        info.workUpper[iVar] /= factor
        info.workRange[iVar] /= factor
        info.workValue[iVar] /= factor
        info.workLowerShift[iVar] /= factor
        info.workUpperShift[iVar] /= factor
    end
    for iRow ∈ 1:num_row
        iVar = e.basis.basicIndex[iRow]
        factor = iVar <= num_col ? col_scale[iVar] : 1.0 / row_scale[iVar - num_col]
        info.baseLower[iRow] *= factor
        info.baseUpper[iRow] *= factor
        info.baseValue[iRow] *= factor
    end
    return e
end

"""
    restore_scale!(e)

Post-solve exit: `unscaleSimplex` followed by unapplying scales from the model
(`moveBackLpAndUnapplyScaling`). The C++ source keeps the workspace scaled and
only removes scales from `HighsLp`; this port restores full state to original units,
yielding identical reported values. Factors remain cached for the next solve
(`considerScaling` reapplies them).
"""
function restore_scale!(e::SimplexEngine)
    e.lp.is_scaled || return e
    unscale_simplex!(e)
    unapply_lp_scale!(e.lp)
    e.nla.scale = nothing
    return e
end
