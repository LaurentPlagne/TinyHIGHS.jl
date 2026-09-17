# `HFactor::rebuild` (HFactorRefactor.cpp:30), complétion des bases déficientes
# (`buildHandleRankDeficiency`/`buildMarkSingC`, HFactor.cpp:1263/1365) et mise à
# jour Forrest-Tomlin (`updateFT`, HFactor.cpp:2300). M1b.

"""`HFactor::rebuild` — rejoue les pivots de `refactor_info` sans Markowitz."""
function rebuild!(f::HFactor)
    luClear!(f)
    f.nwork = 0
    f.basis_matrix_num_el = 0
    stage = f.num_row
    rank_deficiency = 0
    has_pivot = fill(false, f.num_row)
    f.build_synthetic_tick = f.refactor_info.build_synthetic_tick
    info = f.refactor_info

    iK = 1
    while iK <= f.num_row
        iRow = info.pivot_row[iK]
        iVar = info.pivot_var[iK]
        pivot_type = info.pivot_type[iK]
        if pivot_type == kPivotLogical || pivot_type == kPivotUnit
            f.basis_matrix_num_el += 1
            push!(f.l_start, length(f.l_index) + 1)
            push!(f.u_pivot_index, iRow)
            push!(f.u_pivot_value, 1.0)
            push!(f.u_start, length(f.u_index) + 1)
        elseif pivot_type == kPivotRowSingleton || pivot_type == kPivotColSingleton
            start = f.a_start[iVar]
            end_ = f.a_start[iVar + 1] - 1
            pivot_k = 0
            for k ∈ start:end_
                if f.a_index[k] == iRow
                    pivot_k = k
                    break
                end
            end
            abs_pivot = abs(f.a_value[pivot_k])
            if abs_pivot < f.pivot_tolerance
                rank_deficiency = f.nwork + 1
                return rank_deficiency
            end
            if pivot_type == kPivotRowSingleton
                pivot_multiplier = 1.0 / f.a_value[pivot_k]
                for section ∈ 1:2
                    p0 = section == 1 ? start : pivot_k + 1
                    p1 = section == 1 ? pivot_k - 1 : end_
                    for k ∈ p0:p1
                        local_iRow = f.a_index[k]
                        if !has_pivot[local_iRow]
                            push!(f.l_index, local_iRow)
                            push!(f.l_value, f.a_value[k] * pivot_multiplier)
                        else
                            push!(f.u_index, local_iRow)
                            push!(f.u_value, f.a_value[k])
                        end
                    end
                end
                push!(f.l_start, length(f.l_index) + 1)
                push!(f.u_pivot_index, iRow)
                push!(f.u_pivot_value, f.a_value[pivot_k])
                push!(f.u_start, length(f.u_index) + 1)
            else
                for k ∈ start:(pivot_k - 1)
                    push!(f.u_index, f.a_index[k])
                    push!(f.u_value, f.a_value[k])
                end
                for k ∈ (pivot_k + 1):end_
                    push!(f.u_index, f.a_index[k])
                    push!(f.u_value, f.a_value[k])
                end
                push!(f.l_start, length(f.l_index) + 1)
                push!(f.u_pivot_index, iRow)
                push!(f.u_pivot_value, f.a_value[pivot_k])
                push!(f.u_start, length(f.u_index) + 1)
            end
        else
            # Premier pivot de Markowitz : le reste est traité en bloc.
            stage = iK - 1
            break
        end
        f.basic_index[iRow] = iVar
        has_pivot[iRow] = true
        iK += 1
    end

    if stage < f.num_row
        # Complète L par des colonnes identité pour les lignes sans pivot.
        resize!(f.l_start, f.num_row + 1)
        for k ∈ (stage + 1):f.num_row
            f.l_start[k + 1] = f.l_start[k]
        end
        resize!(f.l_pivot_index, f.num_row)
        for k ∈ 1:f.num_row
            f.l_pivot_index[k] = info.pivot_row[k]
        end
        resize!(f.l_pivot_lookup, f.num_row)
        for iRow ∈ 1:f.num_row
            f.l_pivot_lookup[f.l_pivot_index[iRow]] = iRow
        end
        not_in_bump = copy(has_pivot)
        expected_density = 0.0
        column = HVector(f.num_row)
        for k ∈ (stage + 1):f.num_row
            iRow = info.pivot_row[k]
            iVar = info.pivot_var[k]
            clear!(column)
            start = f.a_start[iVar]
            end_ = f.a_start[iVar + 1] - 1
            for iEl ∈ start:end_
                local_iRow = f.a_index[iEl]
                if not_in_bump[local_iRow]
                    push!(f.u_index, local_iRow)
                    push!(f.u_value, f.a_value[iEl])
                else
                    column.count += 1
                    column.index[column.count] = local_iRow
                    column.array[local_iRow] = f.a_value[iEl]
                end
            end
            ftranL!(f, column, expected_density)
            local_density = (1.0 * column.count) / f.num_row
            expected_density = kRunningAverageMultiplier * local_density +
                               (1 - kRunningAverageMultiplier) * expected_density
            tight!(column)
            pivot_k = 0
            for kk ∈ 1:column.count
                if column.index[kk] == iRow
                    pivot_k = kk
                    break
                end
            end
            abs_pivot = abs(column.array[iRow])
            if abs_pivot < f.pivot_tolerance
                rank_deficiency = f.num_row - (k - 1)
                return rank_deficiency
            end
            pivot_multiplier = 1.0 / column.array[iRow]
            for section ∈ 1:2
                p0 = section == 1 ? 1 : pivot_k + 1
                p1 = section == 1 ? pivot_k - 1 : column.count
                for kk ∈ p0:p1
                    local_iRow = column.index[kk]
                    if !has_pivot[local_iRow]
                        push!(f.l_index, local_iRow)
                        push!(f.l_value, column.array[local_iRow] * pivot_multiplier)
                    else
                        push!(f.u_index, local_iRow)
                        push!(f.u_value, column.array[local_iRow])
                    end
                end
            end
            f.l_start[k + 1] = length(f.l_index) + 1
            push!(f.u_pivot_index, iRow)
            push!(f.u_pivot_value, column.array[iRow])
            push!(f.u_start, length(f.u_index) + 1)
            f.basic_index[iRow] = iVar
            has_pivot[iRow] = true
        end
    end
    buildFinish!(f)
    return 0
end

"""
`HFactor::buildHandleRankDeficiency` — remplace les colonnes sans pivot par des
logiques et complète le facteur pour les lignes réelles.
"""
function buildHandleRankDeficiency!(f::HFactor)
    if f.num_basic < f.num_row
        f.rank_deficiency += f.num_row - f.num_basic
    end
    f.row_with_no_pivot = zeros(Int, f.rank_deficiency)
    f.col_with_no_pivot = zeros(Int, f.rank_deficiency)
    lc_rank_deficiency = 0
    resize!(f.iwork, max(f.num_row, f.num_basic))
    fill!(f.iwork, -1)
    for i ∈ 1:f.num_basic
        perm_i = f.permute[i]
        if perm_i > 0
            f.iwork[perm_i] = f.basic_index[i]
        else
            lc_rank_deficiency += 1
            f.col_with_no_pivot[lc_rank_deficiency] = i
        end
    end
    if f.num_basic < f.num_row
        resize!(f.permute, f.num_row)
        for i ∈ (f.num_basic + 1):f.num_row
            lc_rank_deficiency += 1
            f.col_with_no_pivot[lc_rank_deficiency] = i
            f.permute[i] = 0
        end
    end
    lc_rank_deficiency = 0
    for i ∈ 1:f.num_row
        if f.iwork[i] < 0
            lc_rank_deficiency += 1
            f.row_with_no_pivot[lc_rank_deficiency] = i
            f.iwork[i] = -(lc_rank_deficiency)
        end
    end
    if f.num_row < f.num_basic
        for i ∈ (f.num_row + 1):f.num_basic
            lc_rank_deficiency += 1
            f.row_with_no_pivot[lc_rank_deficiency] = i
            f.iwork[i] = -(lc_rank_deficiency)
        end
    end
    row_rank_deficiency = f.rank_deficiency - max(f.num_basic - f.num_row, 0)
    for k ∈ 1:f.rank_deficiency
        iRow = f.row_with_no_pivot[k]
        iCol = f.col_with_no_pivot[k]
        f.permute[iCol] = iRow
        if k <= row_rank_deficiency
            push!(f.l_start, length(f.l_index) + 1)
            push!(f.u_pivot_index, iRow)
            push!(f.u_pivot_value, 1.0)
            push!(f.u_start, length(f.u_index) + 1)
        end
    end
    return f
end

"""`HFactor::buildMarkSingC` — réordonne `basic_index` et note les variables évincées."""
function buildMarkSingC!(f::HFactor)
    basic_index_rank_deficiency = f.rank_deficiency - max(f.num_row - f.num_basic, 0)
    f.var_with_no_pivot = zeros(Int, f.rank_deficiency)
    for k ∈ 1:f.rank_deficiency
        ASMrow = f.row_with_no_pivot[k]
        ASMcol = f.col_with_no_pivot[k]
        f.iwork[ASMrow] = -(ASMcol + 1)
        if ASMcol <= f.num_basic
            f.var_with_no_pivot[k] = f.basic_index[ASMcol]
            f.basic_index[ASMcol] = f.num_col + ASMrow
        elseif f.num_basic < f.num_row
            f.var_with_no_pivot[k] = 0
        end
    end
    return f
end

"""
    update!(f, aq, ep, iRow)

`HFactor::update` — mise à jour Forrest-Tomlin. `aq` porte `B^{-1} a_q` (packé)
et `ep` `B^{-T} e_p` (packé) ; `iRow` est la ligne sortante (1-based).
"""
function update!(f::HFactor, aq::HVector, ep::HVector, iRow::Int)
    clear!(f.refactor_info)
    f.update_method == kUpdateMethodFt ||
        error("seul l'update FT est porté (update_method = $(f.update_method))")
    return updateFT!(f, aq, ep, iRow)
end

"""`HFactor::updateFT` (HFactor.cpp:2300)."""
function updateFT!(f::HFactor, aq::HVector, ep::HVector, iRow::Int)
    p_logic = f.u_pivot_lookup[iRow]
    pivot = f.u_pivot_value[p_logic]
    alpha = aq.array[iRow]
    f.u_pivot_index[p_logic] = 0

    # Supprime la ligne pivot de U.
    for k ∈ f.ur_start[p_logic]:(f.ur_lastp[p_logic] - 1)
        i_logic = f.u_pivot_lookup[f.ur_index[k]]
        f.u_last_p[i_logic] -= 1
        i_last = f.u_last_p[i_logic]
        i_find = f.u_start[i_logic]
        while i_find <= i_last && f.u_index[i_find] != iRow
            i_find += 1
        end
        f.u_index[i_find] = f.u_index[i_last]
        f.u_value[i_find] = f.u_value[i_last]
    end

    # Supprime la colonne pivot de UR.
    for k ∈ f.u_start[p_logic]:(f.u_last_p[p_logic] - 1)
        i_logic = f.u_pivot_lookup[f.u_index[k]]
        f.ur_lastp[i_logic] -= 1
        i_last = f.ur_lastp[i_logic]
        i_find = f.ur_start[i_logic]
        while i_find <= i_last && f.ur_index[i_find] != iRow
            i_find += 1
        end
        f.ur_space[i_logic] += 1
        f.ur_index[i_find] = f.ur_index[i_last]
        f.ur_value[i_find] = f.ur_value[i_last]
    end

    # Nouvelle colonne dans U.
    push!(f.u_start, length(f.u_index) + 1)
    for i ∈ 1:aq.packCount
        if aq.packIndex[i] != iRow
            push!(f.u_index, aq.packIndex[i])
            push!(f.u_value, aq.packValue[i])
        end
    end
    push!(f.u_last_p, length(f.u_index) + 1)
    u_startX = f.u_start[end]
    u_endX = f.u_last_p[end]
    f.u_total_x += u_endX - u_startX + 1

    # Nouveaux éléments UR (avec croissance géométrique).
    for k ∈ u_startX:(u_endX - 1)
        i_logic = f.u_pivot_lookup[f.u_index[k]]
        if f.ur_space[i_logic] == 0
            row_start = f.ur_start[i_logic]
            row_count = f.ur_lastp[i_logic] - row_start
            new_start = length(f.ur_index) + 1
            new_space = trunc(Int, row_count * 1.1 + 5)
            resize!(f.ur_index, new_start + new_space - 1)
            resize!(f.ur_value, new_start + new_space - 1)
            for off ∈ 0:(row_count - 1)
                f.ur_index[new_start + off] = f.ur_index[row_start + off]
                f.ur_value[new_start + off] = f.ur_value[row_start + off]
            end
            f.ur_start[i_logic] = new_start
            f.ur_lastp[i_logic] = new_start + row_count
            f.ur_space[i_logic] = new_space - row_count
        end
        f.ur_space[i_logic] -= 1
        i_put = f.ur_lastp[i_logic]
        f.ur_lastp[i_logic] += 1
        f.ur_index[i_put] = iRow
        f.ur_value[i_put] = f.u_value[k]
    end

    push!(f.ur_start, f.ur_start[p_logic])
    push!(f.ur_lastp, f.ur_start[p_logic])
    push!(f.ur_space,
        f.ur_space[p_logic] + f.ur_lastp[p_logic] - f.ur_start[p_logic])

    f.u_pivot_lookup[iRow] = length(f.u_pivot_index) + 1
    push!(f.u_pivot_index, iRow)
    push!(f.u_pivot_value, pivot * alpha)

    # Ligne `ep` comme matrice R.
    pf_entries_before = f.pf_start[end] - 1
    for i ∈ 1:ep.packCount
        if ep.packIndex[i] != iRow
            push!(f.pf_index, ep.packIndex[i])
            push!(f.pf_value, -ep.packValue[i] * pivot)
        end
    end
    f.u_total_x += length(f.pf_index) - pf_entries_before
    push!(f.pf_pivot_index, iRow)
    push!(f.pf_start, length(f.pf_index) + 1)
    f.u_total_x -= f.u_last_p[p_logic] - f.u_start[p_logic]
    f.u_total_x -= f.ur_lastp[p_logic] - f.ur_start[p_logic]
    return f
end
