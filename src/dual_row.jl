# Port of `simplex/HEkkDualRow.{h,cpp}` (MIT License, HiGHS) — dual ratio
# test (BFRT) and incoming column selection.
#
# Pairs `(iCol, value)` in `workData` follow the source: they are permuted
# in-place then sorted by `iCol` before `updateFlip`.

"""
    DualRow(engine)

Mirror of `HEkkDualRow`: data structures and work buffers for dual ratio test.
"""
mutable struct DualRow
    engine::SimplexEngine
    workSize::Int
    freeList::Set{Int}
    packCount::Int
    packIndex::Vector{Int}
    packValue::Vector{Float64}
    computed_edge_weight::Float64
    workDelta::Float64
    workAlpha::Float64
    workTheta::Float64
    workPivot::Int
    workCount::Int
    workData::Vector{Tuple{Int,Float64}}
    workGroup::Vector{Int}
end

function DualRow(e::SimplexEngine)
    num_tot = e.lp.num_col + e.lp.num_row
    return DualRow(e, num_tot, Set{Int}(), 0, zeros(Int, num_tot),
        zeros(num_tot), 0.0, 0.0, 0.0, 0.0, 0, 0,
        Vector{Tuple{Int,Float64}}(undef, num_tot), Int[])
end

"""`HEkkDualRow::clear`."""
function clear!(d::DualRow)
    d.packCount = 0
    d.workCount = 0
    return d
end

"""Reinitializes a `DualRow` for a new solve — mirrors constructor."""
function reset!(r::DualRow, e::SimplexEngine)
    num_tot = e.lp.num_col + e.lp.num_row
    r.engine = e
    r.workSize = num_tot
    empty!(r.freeList)
    r.packCount = 0
    if length(r.packIndex) != num_tot
        resize!(r.packIndex, num_tot)
        resize!(r.packValue, num_tot)
        resize!(r.workData, num_tot)
    end
    fill!(r.packIndex, 0)
    fill!(r.packValue, 0.0)
    r.computed_edge_weight = 0.0
    r.workDelta = 0.0
    r.workAlpha = 0.0
    r.workTheta = 0.0
    r.workPivot = 0
    r.workCount = 0
    empty!(r.workGroup)
    return r
end

"""`HEkkDualRow::chooseMakepack` — `offset = num_col` for `row_ep`."""
function choose_makepack!(d::DualRow, row::HVector, offset::Int)
    for i ∈ 1:row.count
        index = row.index[i]
        d.packCount += 1
        d.packIndex[d.packCount] = index + offset
        d.packValue[d.packCount] = row.array[index]
    end
    return d
end

"""`HEkkDualRow::choosePossible` — candidats du ratio test."""
function choose_possible!(d::DualRow)
    e = d.engine
    info = e.info
    basis = e.basis
    ta = info.update_count < 10 ? 1e-9 : info.update_count < 20 ? 3e-8 : 1e-6
    td = e.options.dual_feasibility_tolerance
    move_out = d.workDelta < 0 ? -1 : 1
    d.workTheta = kHighsInf
    d.workCount = 0
    for i ∈ 1:d.packCount
        iCol = d.packIndex[i]
        move = basis.nonbasicMove[iCol]
        alpha = d.packValue[i] * move_out * move
        if alpha > ta
            d.workCount += 1
            d.workData[d.workCount] = (iCol, alpha)
            relax = info.workDual[iCol] * move + td
            if d.workTheta * alpha > relax
                d.workTheta = relax / alpha
            end
        end
    end
    return d
end

"""`HEkkDualRow::chooseFinal` — rend `0` si un pivot est choisi, `-1` sinon."""
function choose_final!(d::DualRow)
    e = d.engine
    info = e.info
    basis = e.basis
    # 1. BFRT large step reduction.
    full_count = d.workCount
    d.workCount = 0
    total_change = 0.0
    total_delta = abs(d.workDelta)
    select_theta = 10 * d.workTheta + 1e-7
    while true
        for i ∈ (d.workCount + 1):full_count
            iCol = d.workData[i][1]
            alpha = d.workData[i][2]
            tight = basis.nonbasicMove[iCol] * info.workDual[iCol]
            if alpha * select_theta >= tight
                d.workCount += 1
                d.workData[d.workCount], d.workData[i] =
                    d.workData[i], d.workData[d.workCount]
                total_change += info.workRange[iCol] * alpha
            end
        end
        select_theta *= 10
        (total_change >= total_delta || d.workCount == full_count) && break
    end
    # 2. Quadratic sort of degenerate groups.
    choose_final_work_group_quad!(d) || return -1
    # 3. Selection of largest alpha.
    break_index, break_group = choose_final_large_alpha!(d, d.workCount,
        d.workData, d.workGroup)
    move_out = d.workDelta < 0 ? -1 : 1
    break_index >= 0 || return -1
    d.workPivot = d.workData[break_index][1]
    d.workAlpha = d.workData[break_index][2] * move_out *
                  basis.nonbasicMove[d.workPivot]
    if info.workDual[d.workPivot] * basis.nonbasicMove[d.workPivot] > 0
        d.workTheta = info.workDual[d.workPivot] / d.workAlpha
    else
        d.workTheta = 0.0
    end
    # 4. Variables to flip (BFRT): those in the final group.
    d.workCount = 0
    for i ∈ 1:d.workGroup[break_group + 1]
        iCol = d.workData[i][1]
        move = basis.nonbasicMove[iCol]
        d.workCount += 1
        d.workData[d.workCount] = (iCol, move * info.workRange[iCol])
    end
    if d.workTheta == 0
        d.workCount = 0
    end
    # 5. Column sort (ordered access to A in `updateFlip`).
    sort_work_data!(d.workData, d.workCount)
    return 0
end

"""In-place insertion sort of `workData` over the first `n` elements (zero allocation)."""
function sort_work_data!(workData::Vector{Tuple{Int,Float64}}, n::Int)
    @inbounds for i ∈ 2:n
        key = workData[i]
        j = i - 1
        while j >= 1 && workData[j] > key
            workData[j + 1] = workData[j]
            j -= 1
        end
        workData[j + 1] = key
    end
    return workData
end

"""`HEkkDualRow::chooseFinalWorkGroupQuad`."""
function choose_final_work_group_quad!(d::DualRow)
    e = d.engine
    info = e.info
    basis = e.basis
    td = e.options.dual_feasibility_tolerance
    full_count = d.workCount
    d.workCount = 0
    total_change = kInitialTotalChange
    select_theta = d.workTheta
    total_delta = abs(d.workDelta)
    empty!(d.workGroup)
    push!(d.workGroup, 0)
    prev_work_count = d.workCount
    prev_remain_theta = kInitialRemainTheta
    prev_select_theta = select_theta
    while select_theta < kMaxSelectTheta
        remain_theta = kInitialRemainTheta
        @inbounds for i ∈ (d.workCount + 1):full_count
            iCol = d.workData[i][1]
            value = d.workData[i][2]
            dual = basis.nonbasicMove[iCol] * info.workDual[iCol]
            if dual <= select_theta * value
                d.workCount += 1
                d.workData[d.workCount], d.workData[i] =
                    d.workData[i], d.workData[d.workCount]
                total_change += value * info.workRange[iCol]
            elseif dual + td < remain_theta * value
                remain_theta = (dual + td) / value
            end
        end
        push!(d.workGroup, d.workCount)
        select_theta = remain_theta
        if d.workCount == prev_work_count && prev_select_theta == select_theta &&
           prev_remain_theta == remain_theta
            return false                # loop without progress: CHUZC failure
        end
        prev_work_count = d.workCount
        prev_remain_theta = remain_theta
        prev_select_theta = select_theta
        (total_change >= total_delta || d.workCount == full_count) && break
    end
    return length(d.workGroup) > 1
end

"""
    choose_final_large_alpha!(d, pass_work_count, pass_work_data, pass_work_group)

`HEkkDualRow::chooseFinalLargeAlpha`: last group whose maximum exceeds
`min(0.1 * max, 1)`; ties broken by `numTotPermutation`.
"""
function choose_final_large_alpha!(d::DualRow, pass_work_count::Int,
    pass_work_data::Vector{Tuple{Int,Float64}}, pass_work_group::Vector{Int})
    e = d.engine
    final_compare = 0.0
    for i ∈ 1:pass_work_count
        final_compare = max(final_compare, pass_work_data[i][2])
    end
    final_compare = min(0.1 * final_compare, 1.0)
    count_group = length(pass_work_group) - 1
    break_group = -1
    break_index = -1
    for i_group ∈ (count_group - 1):-1:0
        d_max_final = 0.0
        i_max_final = -1
        for i ∈ (pass_work_group[i_group + 1] + 1):pass_work_group[i_group + 2]
            if d_max_final < pass_work_data[i][2]
                d_max_final = pass_work_data[i][2]
                i_max_final = i
            elseif d_max_final == pass_work_data[i][2]
                jCol = pass_work_data[i_max_final][1]
                iCol = pass_work_data[i][1]
                if e.info.numTotPermutation[iCol] <
                   e.info.numTotPermutation[jCol]
                    i_max_final = i
                end
            end
        end
        if pass_work_data[i_max_final][2] > final_compare
            break_index = i_max_final
            break_group = i_group
            break
        end
    end
    return break_index, break_group
end

"""
    update_flip!(d, bfrtColumn)

`HEkkDualRow::updateFlip`: flips variables to opposite bound and
accumulates the FTRAN-BFRT RHS column.
"""
function update_flip!(d::DualRow, bfrtColumn::HVector)
    e = d.engine
    info = e.info
    dual_objective_value_change = 0.0
    clear!(bfrtColumn)
    for i ∈ 1:d.workCount
        iCol = d.workData[i][1]
        change = d.workData[i][2]
        local_change = change * info.workDual[iCol] * e.cost_scale
        dual_objective_value_change += local_change
        flip_bound!(e, iCol)
        collect_aj!(e.lp.a_matrix, bfrtColumn, iCol, change)
    end
    info.updated_dual_objective_value += dual_objective_value_change
    return d
end

"""`HEkkDualRow::updateDual` — `workDual .-= theta * packValue`."""
function update_dual!(d::DualRow, theta::Float64)
    e = d.engine
    info = e.info
    basis = e.basis
    dual_objective_value_change = 0.0
    for i ∈ 1:d.packCount
        iCol = d.packIndex[i]
        info.workDual[iCol] -= theta * d.packValue[i]
        delta_dual = theta * d.packValue[i]
        local_change = basis.nonbasicFlag[iCol] *
                       (-info.workValue[iCol] * delta_dual) * e.cost_scale
        dual_objective_value_change += local_change
    end
    info.updated_dual_objective_value += dual_objective_value_change
    return d
end

"""`HEkkDualRow::createFreelist` — free nonbasic columns."""
function create_freelist!(d::DualRow)
    e = d.engine
    empty!(d.freeList)
    for iVar ∈ 1:(e.lp.num_col + e.lp.num_row)
        if e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue &&
           e.info.workLower[iVar] == -kHighsInf &&
           e.info.workUpper[iVar] == kHighsInf
            push!(d.freeList, iVar)
        end
    end
    return d
end

"""`HEkkDualRow::createFreemove` — temporary movement of free columns."""
function create_freemove!(d::DualRow, row_ep::HVector)
    isempty(d.freeList) && return d
    e = d.engine
    ta = e.info.update_count < 10 ? 1e-9 :
         e.info.update_count < 20 ? 3e-8 : 1e-6
    move_out = d.workDelta < 0 ? -1 : 1
    for iVar ∈ d.freeList
        alpha = compute_dot(e.lp.a_matrix, row_ep.array, iVar)
        if abs(alpha) > ta
            e.basis.nonbasicMove[iVar] =
                alpha * move_out > 0 ? Int8(1) : Int8(-1)
        end
    end
    return d
end

"""`HEkkDualRow::deleteFreemove`."""
function delete_freemove!(d::DualRow)
    isempty(d.freeList) && return d
    for iVar ∈ d.freeList
        d.engine.basis.nonbasicMove[iVar] = kNonbasicMoveZe
    end
    return d
end

"""`HEkkDualRow::deleteFreelist`."""
function delete_freelist!(d::DualRow, iVar::Int)
    isempty(d.freeList) || delete!(d.freeList, iVar)
    return d
end

"""
`HEkkDualRow::computeDevexWeight`: exact Devex weight of the pivot row, sum
of squared pack entries over the reference framework (`devex_index`).
"""
function compute_devex_weight!(d::DualRow)
    e = d.engine
    d.computed_edge_weight = 0.0
    for el ∈ 1:d.packCount
        vr = d.packIndex[el]
        e.basis.nonbasicFlag[vr] == kNonbasicFlagFalse && continue
        pv = e.info.devex_index[vr] * d.packValue[el]
        if pv != 0.0
            d.computed_edge_weight += pv * pv
        end
    end
    return d.computed_edge_weight
end
