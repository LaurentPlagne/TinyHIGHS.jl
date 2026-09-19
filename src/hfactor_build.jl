# `HFactor::buildSimple`, `HFactor::buildKernel`, `HFactor::buildFinish`
# (HFactor.cpp:560, 871, 1400). Full rank deficiency handling and `refactor_info_`
# are covered in M1b (cf. header of hfactor.jl).

"""`HFactor::buildSimple` — unit columns, singletons, then kernel matrix."""
function buildSimple!(f::HFactor)
    luClear!(f)
    resize!(f.permute, f.num_basic)
    fill!(f.permute, 0)
    fill!(f.mr_count_before, 0)
    f.nwork = 0
    fill!(f.iwork, 0)

    Bcount = 0
    for iCol ∈ 1:f.num_basic
        iMat = f.basic_index[iCol]
        iRow = 0
        pivot_type = kPivotIllegal
        if iMat > f.num_col
            # Logical column.
            lc_iRow = iMat - f.num_col
            if f.mr_count_before[lc_iRow] >= 0
                iRow = lc_iRow
                pivot_type = kPivotLogical
            else
                f.mr_count_before[lc_iRow] += 1
                Bcount += 1
                f.b_index[Bcount] = lc_iRow
                f.b_value[Bcount] = 1.0
                f.nwork += 1
                f.iwork[f.nwork] = iCol
            end
        else
            start = f.a_start[iMat]
            count = f.a_start[iMat + 1] - start
            ok_unit_col = count == 1 && f.a_value[start] == 1.0 &&
                          f.mr_count_before[f.a_index[start]] >= 0
            if ok_unit_col
                iRow = f.a_index[start]
                pivot_type = kPivotColSingleton
            else
                for k ∈ start:(start + count - 1)
                    f.mr_count_before[f.a_index[k]] += 1
                    Bcount += 1
                    f.b_index[Bcount] = f.a_index[k]
                    f.b_value[Bcount] = f.a_value[k]
                end
                f.nwork += 1
                f.iwork[f.nwork] = iCol
            end
        end
        if iRow > 0
            f.permute[iCol] = iRow
            push!(f.l_start, length(f.l_index) + 1)
            push!(f.u_pivot_index, iRow)
            push!(f.u_pivot_value, 1.0)
            push!(f.u_start, length(f.u_index) + 1)
            f.mr_count_before[iRow] = -f.num_basic
            push!(f.refactor_info.pivot_row, iRow)
            push!(f.refactor_info.pivot_var, f.basic_index[iCol])
            push!(f.refactor_info.pivot_type, pivot_type)
        end
        f.b_start[iCol + 1] = Bcount + 1
        f.b_var[iCol] = iMat
    end
    f.basis_matrix_num_el = f.num_row - f.nwork + Bcount
    f.build_synthetic_tick += Bcount * 60 + (f.num_row - f.nwork) * 80

    # Singleton search in successive passes.
    while f.nwork > 0
        nworkLast = f.nwork
        f.nwork = 0
        for i ∈ 1:nworkLast
            iCol = f.iwork[i]
            start = f.b_start[iCol]
            end_ = f.b_start[iCol + 1] - 1
            pivot_k = 0
            found_row_singleton = false
            count = 0
            for k ∈ start:end_
                iRow = f.b_index[k]
                if f.mr_count_before[iRow] == 1
                    pivot_k = k
                    found_row_singleton = true
                    break
                end
                if f.mr_count_before[iRow] > 1
                    pivot_k = k
                    count += 1
                end
            end
            if found_row_singleton
                pivot_multiplier = 1.0 / f.b_value[pivot_k]
                for section ∈ 1:2
                    p0 = section == 1 ? start : pivot_k + 1
                    p1 = section == 1 ? pivot_k - 1 : end_
                    for k ∈ p0:p1
                        iRow = f.b_index[k]
                        if f.mr_count_before[iRow] > 0
                            push!(f.l_index, iRow)
                            push!(f.l_value, f.b_value[k] * pivot_multiplier)
                        else
                            push!(f.u_index, iRow)
                            push!(f.u_value, f.b_value[k])
                        end
                        f.mr_count_before[iRow] -= 1
                    end
                end
                iRow = f.b_index[pivot_k]
                f.mr_count_before[iRow] = 0
                f.permute[iCol] = iRow
                push!(f.l_start, length(f.l_index) + 1)
                push!(f.u_pivot_index, iRow)
                push!(f.u_pivot_value, f.b_value[pivot_k])
                push!(f.u_start, length(f.u_index) + 1)
                push!(f.refactor_info.pivot_row, iRow)
                push!(f.refactor_info.pivot_var, f.basic_index[iCol])
                push!(f.refactor_info.pivot_type, kPivotRowSingleton)
            elseif count == 1
                for k ∈ start:(pivot_k - 1)
                    push!(f.u_index, f.b_index[k])
                    push!(f.u_value, f.b_value[k])
                end
                for k ∈ (pivot_k + 1):end_
                    push!(f.u_index, f.b_index[k])
                    push!(f.u_value, f.b_value[k])
                end
                iRow = f.b_index[pivot_k]
                f.mr_count_before[iRow] = 0
                f.permute[iCol] = iRow
                push!(f.l_start, length(f.l_index) + 1)
                push!(f.u_pivot_index, iRow)
                push!(f.u_pivot_value, f.b_value[pivot_k])
                push!(f.u_start, length(f.u_index) + 1)
                push!(f.refactor_info.pivot_row, iRow)
                push!(f.refactor_info.pivot_var, f.basic_index[iCol])
                push!(f.refactor_info.pivot_type, kPivotColSingleton)
            else
                f.nwork += 1
                f.iwork[f.nwork] = iCol
            end
        end
        nworkLast == f.nwork && break
    end

    # Kernel preparation: rows, then columns.
    fill!(f.row_link_first, -1)
    fill!(f.mr_count, 0)
    mr_countX = 0
    f.kernel_num_el = 0
    for iRow ∈ 1:f.num_row
        count = f.mr_count_before[iRow]
        if count > 0
            f.mr_start[iRow] = mr_countX + 1
            f.mr_space[iRow] = count * 2
            mr_countX += count * 2
            rlinkAdd!(f, iRow, count)
            f.kernel_num_el += count + 1
        end
    end
    resize!(f.mr_index, mr_countX)
    fill!(f.mr_index, 0)

    fill!(f.col_link_first, -1)
    fill!(f.mc_count_a, 0)
    fill!(f.mc_count_n, 0)
    empty!(f.mc_index)
    empty!(f.mc_value)
    MCcountX = 0
    for i ∈ 1:f.nwork
        iCol = f.iwork[i]
        f.mc_var[iCol] = f.b_var[iCol]
        f.mc_start[iCol] = MCcountX + 1
        f.mc_space[iCol] = (f.b_start[iCol + 1] - f.b_start[iCol]) * 2
        MCcountX += f.mc_space[iCol]
        resize!(f.mc_index, MCcountX)
        resize!(f.mc_value, MCcountX)
        for k ∈ f.b_start[iCol]:(f.b_start[iCol + 1] - 1)
            iRow = f.b_index[k]
            value = f.b_value[k]
            if f.mr_count_before[iRow] > 0
                colInsert!(f, iCol, iRow, value)
                rowInsert!(f, iCol, iRow)
            else
                colStoreN!(f, iCol, iRow, value)
            end
        end
        colFixMax!(f, iCol)
        clinkAdd!(f, iCol, f.mc_count_a[iCol])
    end
    f.build_synthetic_tick += (f.num_row + f.nwork + MCcountX) * 40 + mr_countX * 20
    f.kernel_dim = f.nwork
    return f
end

"""`HFactor::buildKernel` — Markowitz pivot search and elimination."""
function buildKernel!(f::HFactor)
    while f.nwork > 0
        f.nwork -= 1

        # 1. Pivot search.
        jColPivot = 0
        iRowPivot = 0
        searchLimit = min(f.nwork, 8)
        searchCount = 0
        merit_limit = 1.0 * f.num_basic * f.num_row
        merit_pivot = merit_limit
        foundPivot = false
        if f.col_link_first[2] != -1
            jColPivot = f.col_link_first[2]
            iRowPivot = f.mc_index[f.mc_start[jColPivot]]
            foundPivot = true
        elseif f.row_link_first[2] != -1
            iRowPivot = f.row_link_first[2]
            jColPivot = f.mr_index[f.mr_start[iRowPivot]]
            foundPivot = true
        end
        singleton_pivot = foundPivot
        max_count = max(f.num_row, f.num_basic)
        count = 2
        while !foundPivot && count <= max_count
            if count <= f.num_row
                j = f.col_link_first[count + 1]
                while j != -1
                    min_pivot = f.mc_min_pivot[j]
                    start = f.mc_start[j]
                    k = start
                    stop = start + f.mc_count_a[j] - 1
                    while k <= stop
                        if abs(f.mc_value[k]) >= min_pivot
                            i = f.mc_index[k]
                            row_count = f.mr_count[i]
                            merit_local = 1.0 * (count - 1) * (row_count - 1)
                            if merit_pivot > merit_local
                                merit_pivot = merit_local
                                jColPivot = j
                                iRowPivot = i
                                foundPivot = foundPivot || (row_count < count)
                            end
                        end
                        k += 1
                    end
                    if searchCount >= searchLimit && merit_pivot < merit_limit
                        foundPivot = true
                    end
                    searchCount += 1
                    foundPivot && break
                    j = f.col_link_next[j]
                end
            end
            if count <= f.num_basic && !foundPivot
                i = f.row_link_first[count + 1]
                while i != -1
                    start = f.mr_start[i]
                    k = start
                    stop = start + f.mr_count[i] - 1
                    while k <= stop
                        j = f.mr_index[k]
                        column_count = f.mc_count_a[j]
                        merit_local = 1.0 * (count - 1) * (column_count - 1)
                        if merit_local < merit_pivot
                            ifind = f.mc_start[j]
                            while f.mc_index[ifind] != i
                                ifind += 1
                            end
                            if abs(f.mc_value[ifind]) >= f.mc_min_pivot[j]
                                merit_pivot = merit_local
                                jColPivot = j
                                iRowPivot = i
                                foundPivot = foundPivot || (column_count <= count)
                            end
                        end
                        k += 1
                    end
                    if searchCount >= searchLimit && merit_pivot < merit_limit
                        foundPivot = true
                    end
                    searchCount += 1
                    foundPivot && break
                    i = f.row_link_next[i]
                end
            end
            count += 1
        end

        if iRowPivot <= 0
            f.rank_deficiency = f.nwork + 1
            return f.rank_deficiency
        end

        # 2. Elimination by pivot.
        pivot_multiplier = colDelete!(f, jColPivot, iRowPivot)
        rowDelete!(f, jColPivot, iRowPivot)
        clinkDel!(f, jColPivot)
        rlinkDel!(f, iRowPivot)
        if abs(pivot_multiplier) < f.pivot_tolerance
            # Deferred singular pivot: other valid pivots may exist.
            if f.mr_count[iRowPivot] == 0
                clinkAdd!(f, jColPivot, f.mc_count_a[jColPivot])
            else
                zeroCol!(f, jColPivot)
                rlinkAdd!(f, iRowPivot, f.mr_count[iRowPivot])
            end
            f.nwork += 1
            continue
        end
        f.permute[jColPivot] = iRowPivot
        push!(f.refactor_info.pivot_row, iRowPivot)
        push!(f.refactor_info.pivot_var, f.basic_index[jColPivot])
        push!(f.refactor_info.pivot_type, kPivotMarkowitz)

        # 2.2. Active pivot column -> L.
        start_A = f.mc_start[jColPivot]
        end_A = start_A + f.mc_count_a[jColPivot] - 1
        mwz_column_count = 0
        for k ∈ start_A:end_A
            iRow = f.mc_index[k]
            value = f.mc_value[k] / pivot_multiplier
            mwz_column_count += 1
            f.mwz_column_index[mwz_column_count] = iRow
            f.mwz_column_array[iRow] = value
            f.mwz_column_mark[iRow] = true
            push!(f.l_index, iRow)
            push!(f.l_value, value)
            f.mr_count_before[iRow] = f.mr_count[iRow]
            rowDelete!(f, jColPivot, iRow)
        end
        push!(f.l_start, length(f.l_index) + 1)

        # 2.3. Non-active pivot column -> U.
        end_N = start_A + f.mc_space[jColPivot] - 1
        start_N = end_N - f.mc_count_n[jColPivot] + 1
        for i ∈ start_N:end_N
            push!(f.u_index, f.mc_index[i])
            push!(f.u_value, f.mc_value[i])
        end
        push!(f.u_pivot_index, iRowPivot)
        push!(f.u_pivot_value, pivot_multiplier)
        push!(f.u_start, length(f.u_index) + 1)

        # 2.4. Elimination on other columns of pivot row.
        row_start = f.mr_start[iRowPivot]
        row_end = row_start + f.mr_count[iRowPivot] - 1
        for row_k ∈ row_start:row_end
            iCol = f.mr_index[row_k]
            my_count = f.mc_count_a[iCol]
            my_start = f.mc_start[iCol]
            my_end = my_start + my_count - 2
            my_pivot = colDelete!(f, iCol, iRowPivot)
            colStoreN!(f, iCol, iRowPivot, my_pivot)

            nFillin = mwz_column_count
            nCancel = 0
            for my_k ∈ my_start:my_end
                iRow = f.mc_index[my_k]
                value = f.mc_value[my_k]
                if f.mwz_column_mark[iRow]
                    f.mwz_column_mark[iRow] = false
                    nFillin -= 1
                    value -= my_pivot * f.mwz_column_array[iRow]
                    if abs(value) < kHighsTiny
                        value = 0.0
                        nCancel += 1
                    end
                    f.mc_value[my_k] = value
                end
            end

            if nCancel > 0
                new_end = my_start
                for my_k ∈ my_start:my_end
                    if f.mc_value[my_k] != 0.0
                        f.mc_index[new_end] = f.mc_index[my_k]
                        f.mc_value[new_end] = f.mc_value[my_k]
                        new_end += 1
                    else
                        rowDelete!(f, iCol, f.mc_index[my_k])
                    end
                end
                f.mc_count_a[iCol] = new_end - my_start
            end

            if nFillin > 0
                if f.mc_count_a[iCol] + f.mc_count_n[iCol] + nFillin > f.mc_space[iCol]
                    p1 = f.mc_start[iCol]
                    p2 = p1 + f.mc_count_a[iCol]
                    p3 = p1 + f.mc_space[iCol] - f.mc_count_n[iCol]
                    p4 = p1 + f.mc_space[iCol]
                    f.mc_space[iCol] += max(f.mc_space[iCol], nFillin)
                    p5 = length(f.mc_index) + 1
                    f.mc_start[iCol] = p5
                    p7 = p5 + f.mc_space[iCol] - f.mc_count_n[iCol]
                    resize!(f.mc_index, p5 + f.mc_space[iCol] - 1)
                    resize!(f.mc_value, p5 + f.mc_space[iCol] - 1)
                    for off ∈ 0:(p2 - p1 - 1)
                        f.mc_index[p5 + off] = f.mc_index[p1 + off]
                        f.mc_value[p5 + off] = f.mc_value[p1 + off]
                    end
                    for off ∈ 0:(p4 - p3 - 1)
                        f.mc_index[p7 + off] = f.mc_index[p3 + off]
                        f.mc_value[p7 + off] = f.mc_value[p3 + off]
                    end
                end
                for i ∈ 1:mwz_column_count
                    iRow = f.mwz_column_index[i]
                    if f.mwz_column_mark[iRow]
                        colInsert!(f, iCol, iRow, -my_pivot * f.mwz_column_array[iRow])
                    end
                end
                for i ∈ 1:mwz_column_count
                    iRow = f.mwz_column_index[i]
                    if f.mwz_column_mark[iRow]
                        if f.mr_count[iRow] == f.mr_space[iRow]
                            p1 = f.mr_start[iRow]
                            p2 = p1 + f.mr_count[iRow]
                            p3 = length(f.mr_index) + 1
                            f.mr_start[iRow] = p3
                            f.mr_space[iRow] *= 2
                            resize!(f.mr_index, p3 + f.mr_space[iRow] - 1)
                            for off ∈ 0:(p2 - p1 - 1)
                                f.mr_index[p3 + off] = f.mr_index[p1 + off]
                            end
                        end
                        rowInsert!(f, iCol, iRow)
                    end
                end
            end

            for i ∈ 1:mwz_column_count
                f.mwz_column_mark[f.mwz_column_index[i]] = true
            end

            colFixMax!(f, iCol)
            if my_count != f.mc_count_a[iCol]
                clinkDel!(f, iCol)
                clinkAdd!(f, iCol, f.mc_count_a[iCol])
            end
        end

        for i ∈ 1:mwz_column_count
            f.mwz_column_mark[f.mwz_column_index[i]] = false
        end

        for i ∈ start_A:end_A
            iRow = f.mc_index[i]
            if f.mr_count_before[iRow] != f.mr_count[iRow]
                rlinkDel!(f, iRow)
                rlinkAdd!(f, iRow, f.mr_count[iRow])
            end
        end
    end
    f.rank_deficiency = 0
    return 0
end

"""`HFactor::buildFinish` — U row-wise, LR/UR, `basic_index` permutation."""
function buildFinish!(f::HFactor)
    for i ∈ 1:f.num_row
        f.u_pivot_lookup[f.u_pivot_index[i]] = i
    end
    copyto!(f.l_pivot_index, f.u_pivot_index)
    copyto!(f.l_pivot_lookup, f.u_pivot_lookup)

    LcountX = length(f.l_index)
    resize!(f.lr_index, LcountX)
    resize!(f.lr_value, LcountX)
    resize!(f.iwork, f.num_row)
    fill!(f.iwork, 0)
    for k ∈ 1:LcountX
        f.iwork[f.l_pivot_lookup[f.l_index[k]]] += 1
    end
    resize!(f.lr_start, f.num_row + 1)
    f.lr_start[1] = 1
    for i ∈ 1:f.num_row
        f.lr_start[i + 1] = f.lr_start[i] + f.iwork[i]
    end
    copyto!(f.iwork, 1, f.lr_start, 1, f.num_row)
    for i ∈ 1:f.num_row
        index = f.l_pivot_index[i]
        for k ∈ f.l_start[i]:(f.l_start[i + 1] - 1)
            iRow = f.l_pivot_lookup[f.l_index[k]]
            i_put = f.iwork[iRow]
            f.iwork[iRow] += 1
            f.lr_index[i_put] = index
            f.lr_value[i_put] = f.l_value[k]
        end
    end

    push!(f.u_start, 1)
    resize!(f.u_last_p, f.num_row)
    copyto!(f.u_last_p, 1, f.u_start, 2, f.num_row)
    resize!(f.u_start, f.num_row)

    u_countX = length(f.u_index)
    ur_stuff_size = f.update_method == kUpdateMethodFt ? 5 : 0
    resize!(f.ur_index, u_countX + ur_stuff_size * f.num_row)
    resize!(f.ur_value, u_countX + ur_stuff_size * f.num_row)
    resize!(f.ur_start, f.num_row + 1)
    resize!(f.ur_lastp, f.num_row)
    fill!(f.ur_lastp, 0)
    resize!(f.ur_space, f.num_row)
    fill!(f.ur_space, ur_stuff_size)
    for k ∈ 1:u_countX
        f.ur_lastp[f.u_pivot_lookup[f.u_index[k]]] += 1
    end
    f.ur_start[1] = 1
    for i ∈ 1:f.num_row
        f.ur_start[i + 1] = f.ur_start[i] + f.ur_lastp[i] + ur_stuff_size
    end
    resize!(f.ur_start, f.num_row)
    for i ∈ 1:f.num_row
        f.ur_lastp[i] = f.ur_start[i]
    end
    for i ∈ 1:f.num_row
        index = f.u_pivot_index[i]
        for k ∈ f.u_start[i]:(f.u_last_p[i] - 1)
            iRow = f.u_pivot_lookup[f.u_index[k]]
            i_put = f.ur_lastp[iRow]
            f.ur_lastp[iRow] += 1
            f.ur_index[i_put] = index
            f.ur_value[i_put] = f.u_value[k]
        end
    end

    # Refactorization merits (HFactor.cpp:1475).
    f.u_merit_x = trunc(Int, f.num_row + (LcountX + u_countX) * 1.5)
    f.u_total_x = u_countX
    if f.update_method == 2  # kUpdateMethodPf
        f.u_merit_x = f.num_row + u_countX * 4
    elseif f.update_method == 3  # kUpdateMethodMpf
        f.u_merit_x = f.num_row + u_countX * 3
    end

    empty!(f.pf_pivot_value)
    empty!(f.pf_pivot_index)
    empty!(f.pf_start)
    push!(f.pf_start, 1)
    empty!(f.pf_index)
    empty!(f.pf_value)

    # Permutation of `basic_index`: variable originally at position i moves to
    # position permute[i] (HFactor.cpp:1491). Not during rebuild:
    # in that case `basic_index` is already row-ordered and `refactor_info.use` is set.
    if !f.refactor_info.use
        resize!(f.iwork, f.num_basic)
        copyto!(f.iwork, f.basic_index)
        @inbounds for i ∈ 1:f.num_basic
            f.basic_index[f.permute[i]] = f.iwork[i]
        end
        f.build_synthetic_tick += f.num_row * 80 +
                                  (length(f.l_index) + length(f.u_index)) * 60
    end

    # SIMD vectorized pre-inversion of U diagonal pivots
    resize!(f.u_pivot_inv_value, length(f.u_pivot_value))
    @inbounds @simd for i ∈ eachindex(f.u_pivot_value)
        f.u_pivot_inv_value[i] = 1.0 / f.u_pivot_value[i]
    end

    return f
end
