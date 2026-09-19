# Inline helper functions of `HFactor.h` (`zeroCol`/`luClear` in HFactor.cpp).
# Stored positions are 1-based (cf. header of hfactor.jl).

"""`HFactor::luClear`: clears L and U factors (l_start/u_start reset to 1)."""
function luClear!(f::HFactor)
    empty!(f.l_start)
    push!(f.l_start, 1)
    empty!(f.l_index)
    empty!(f.l_value)
    empty!(f.u_pivot_index)
    empty!(f.u_pivot_value)
    empty!(f.u_pivot_inv_value)
    empty!(f.u_start)
    push!(f.u_start, 1)
    empty!(f.u_index)
    empty!(f.u_value)
    return f
end

"""`HFactor::colInsert` — inserts an active element at the end of active column section."""
function colInsert!(f::HFactor, iCol::Int, iRow::Int, value::Float64)
    iput = f.mc_start[iCol] + f.mc_count_a[iCol]
    f.mc_count_a[iCol] += 1
    f.mc_index[iput] = iRow
    f.mc_value[iput] = value
    return nothing
end

"""`HFactor::colStoreN` — inserts an inactive element from the column end."""
function colStoreN!(f::HFactor, iCol::Int, iRow::Int, value::Float64)
    f.mc_count_n[iCol] += 1
    iput = f.mc_start[iCol] + f.mc_space[iCol] - f.mc_count_n[iCol]
    f.mc_index[iput] = iRow
    f.mc_value[iput] = value
    return nothing
end

"""`HFactor::colFixMax` — largest |active value| * pivot threshold."""
function colFixMax!(f::HFactor, iCol::Int)
    max_value = 0.0
    for k ∈ f.mc_start[iCol]:(f.mc_start[iCol] + f.mc_count_a[iCol] - 1)
        max_value = max(max_value, abs(f.mc_value[k]))
    end
    f.mc_min_pivot[iCol] = max_value * f.pivot_threshold
    return nothing
end

"""`HFactor::colDelete` — removes `iRow` from column (swaps with end)."""
function colDelete!(f::HFactor, iCol::Int, iRow::Int)
    f.mc_count_a[iCol] -= 1
    imov = f.mc_start[iCol] + f.mc_count_a[iCol]
    idel = f.mc_start[iCol]
    while f.mc_index[idel] != iRow
        idel += 1
    end
    pivot_multiplier = f.mc_value[idel]
    f.mc_index[idel] = f.mc_index[imov]
    f.mc_value[idel] = f.mc_value[imov]
    return pivot_multiplier
end

"""`HFactor::rowInsert` — appends column to row end."""
function rowInsert!(f::HFactor, iCol::Int, iRow::Int)
    iput = f.mr_start[iRow] + f.mr_count[iRow]
    f.mr_count[iRow] += 1
    f.mr_index[iput] = iCol
    return nothing
end

"""`HFactor::rowDelete` — removes `iCol` from row (swaps with end)."""
function rowDelete!(f::HFactor, iCol::Int, iRow::Int)
    f.mr_count[iRow] -= 1
    imov = f.mr_start[iRow] + f.mr_count[iRow]
    idel = f.mr_start[iRow]
    while f.mr_index[idel] != iCol
        idel += 1
    end
    f.mr_index[idel] = f.mr_index[imov]
    return nothing
end

# Doubly-linked bucket lists by element count. `col_link_last[index] = -2 - count`
# encodes bucket index (negative value); links are 1-based, 0 = absent (upstream `-1`).

"""`HFactor::clinkAdd`."""
function clinkAdd!(f::HFactor, index::Int, count::Int)
    mover = f.col_link_first[count + 1]
    f.col_link_last[index] = -2 - count
    f.col_link_next[index] = mover
    f.col_link_first[count + 1] = index
    mover > 0 && (f.col_link_last[mover] = index)
    return nothing
end

"""`HFactor::clinkDel`."""
function clinkDel!(f::HFactor, index::Int)
    xlast = f.col_link_last[index]
    xnext = f.col_link_next[index]
    if xlast > 0
        f.col_link_next[xlast] = xnext
    else
        f.col_link_first[-xlast - 1] = xnext
    end
    xnext > 0 && (f.col_link_last[xnext] = xlast)
    return nothing
end

"""`HFactor::rlinkAdd`."""
function rlinkAdd!(f::HFactor, index::Int, count::Int)
    mover = f.row_link_first[count + 1]
    f.row_link_last[index] = -2 - count
    f.row_link_next[index] = mover
    f.row_link_first[count + 1] = index
    mover > 0 && (f.row_link_last[mover] = index)
    return nothing
end

"""`HFactor::rlinkDel`."""
function rlinkDel!(f::HFactor, index::Int)
    xlast = f.row_link_last[index]
    xnext = f.row_link_next[index]
    if xlast > 0
        f.row_link_next[xlast] = xnext
    else
        f.row_link_first[-xlast - 1] = xnext
    end
    xnext > 0 && (f.row_link_last[xnext] = xlast)
    return nothing
end

"""`HFactor::zeroCol` — neutralizes a column (deferred singular pivot)."""
function zeroCol!(f::HFactor, jCol::Int)
    a_start = f.mc_start[jCol]
    a_end = a_start + f.mc_count_a[jCol] - 1
    for iEl ∈ a_start:a_end
        iRow = f.mc_index[iEl]
        rowDelete!(f, jCol, iRow)
        rlinkDel!(f, iRow)
        rlinkAdd!(f, iRow, f.mr_count[iRow])
    end
    clinkDel!(f, jCol)
    f.mc_count_a[jCol] = 0
    f.mc_count_n[jCol] = 0
    return nothing
end
