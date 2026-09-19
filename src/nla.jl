# Partial port of `simplex/HSimplexNla.{h,cpp}` (M2b): solves in scaled space
# and scaling application around `HFactor`.
#
# Not ported: `ProductFormUpdate` (PF updates), `SimplexIterate`,
# `putInvert`/`getInvert`, `addCols`/`addRows`, telemetry/reporting.
# Factors are provided (or `nothing` = unscaled).
#
# 1-based convention: `basic_index[i]` is the variable number (1-based) of
# the i-th basis column; scale factors follow the same convention
# as `HighsScale` (variable ≤ num_col → `col`, otherwise slack → `row`).

const kDensityForIndexing = 0.4

"""
    Nla(factor, num_col, num_row; scale = nothing)

Numerical Linear Algebra (NLA) abstraction layer over `HFactor`, equivalent to HiGHS's `HSimplexNla`.

Coordinates scaled-space and original-space transformations, managing row and column
scaling factors during forward transformation (FTRAN, \$B x = b\$) and backward
transformation (BTRAN, \$B^T y = c\$).
"""
mutable struct Nla
    num_col::Int
    num_row::Int
    basic_index::Vector{Int}
    scale::Union{Nothing,Scale}
    factor::HFactor
    build_synthetic_tick::Float64
end

"""
Unlike `HFactor`, NLA **shares** `factor.basic_index` (in C++, the class
stores the caller's pointer): `build!` permutes the array in-place
and scale factors remain aligned with basis column order.
"""
function Nla(factor::HFactor, num_col::Int, num_row::Int;
    scale::Union{Nothing,Scale}=nothing)
    return Nla(num_col, num_row, factor.basic_index, scale, factor, 0.0)
end

"""`HSimplexNla::sparseLoopStyle` — `(use_indices, to_entry)`."""
function sparse_loop_style(count::Int, dim::Int)
    use_indices = count >= 0 && count < kDensityForIndexing * dim
    return use_indices, (use_indices ? count : dim)
end

"""`HSimplexNla::variableScaleFactor`."""
function variable_scale_factor(nla::Nla, iVar::Int)
    nla.scale === nothing && return 1.0
    return iVar <= nla.num_col ? nla.scale.col[iVar] :
           1.0 / nla.scale.row[iVar - nla.num_col]
end

"""`HSimplexNla::basicColScaleFactor` — `iCol` indexes the basis (1-based)."""
function basic_col_scale_factor(nla::Nla, iCol::Int)
    nla.scale === nothing && return 1.0
    return variable_scale_factor(nla, nla.basic_index[iCol])
end

"""`HSimplexNla::pivotInScaledSpace`."""
function pivot_in_scaled_space(nla::Nla, aq::HVector, variable_in::Int,
    row_out::Int)
    return aq.array[row_out] * variable_scale_factor(nla, variable_in) /
           variable_scale_factor(nla, nla.basic_index[row_out])
end

"""`HSimplexNla::invert` — factor reconstruction, preserved tick."""
function invert!(nla::Nla)
    rank_deficiency = build!(nla.factor)
    nla.build_synthetic_tick = nla.factor.build_synthetic_tick
    return rank_deficiency
end

"""`HSimplexNla::ftranInScaledSpace` / `btranInScaledSpace`."""
ftran_in_scaled_space!(nla::Nla, rhs::HVector, expected_density::Float64) =
    ftranCall!(nla.factor, rhs, expected_density)
btran_in_scaled_space!(nla::Nla, rhs::HVector, expected_density::Float64) =
    btranCall!(nla.factor, rhs, expected_density)

"""`HSimplexNla::applyBasisMatrixRowScale`."""
function apply_basis_matrix_row_scale!(nla::Nla, rhs::HVector)
    nla.scale === nothing && return rhs
    row_scale = nla.scale.row
    use_indices, to_entry = sparse_loop_style(rhs.count, nla.num_row)
    for iEntry ∈ 1:to_entry
        iRow = use_indices ? rhs.index[iEntry] : iEntry
        rhs.array[iRow] *= row_scale[iRow]
    end
    return rhs
end

"""`HSimplexNla::applyBasisMatrixColScale`."""
function apply_basis_matrix_col_scale!(nla::Nla, rhs::HVector)
    nla.scale === nothing && return rhs
    col_scale = nla.scale.col
    row_scale = nla.scale.row
    use_indices, to_entry = sparse_loop_style(rhs.count, nla.num_row)
    for iEntry ∈ 1:to_entry
        iCol = use_indices ? rhs.index[iEntry] : iEntry
        iVar = nla.basic_index[iCol]
        if iVar <= nla.num_col
            rhs.array[iCol] *= col_scale[iVar]
        else
            rhs.array[iCol] /= row_scale[iVar - nla.num_col]
        end
    end
    return rhs
end

"""`HSimplexNla::unapplyBasisMatrixRowScale`."""
function unapply_basis_matrix_row_scale!(nla::Nla, rhs::HVector)
    nla.scale === nothing && return rhs
    row_scale = nla.scale.row
    use_indices, to_entry = sparse_loop_style(rhs.count, nla.num_row)
    for iEntry ∈ 1:to_entry
        iRow = use_indices ? rhs.index[iEntry] : iEntry
        rhs.array[iRow] /= row_scale[iRow]
    end
    return rhs
end

"""
    ftran!(nla::Nla, rhs::HVector, expected_density::Float64)

Perform forward transformation \$B x = b\$ with basis scaling transformations applied.
Updates `rhs` in-place.
"""
function ftran!(nla::Nla, rhs::HVector, expected_density::Float64)
    apply_basis_matrix_row_scale!(nla, rhs)
    ftran_in_scaled_space!(nla, rhs, expected_density)
    apply_basis_matrix_col_scale!(nla, rhs)
    return rhs
end

"""
    btran!(nla::Nla, rhs::HVector, expected_density::Float64)

Perform backward transformation \$B^T y = c\$ with basis scaling transformations applied.
Updates `rhs` in-place.
"""
function btran!(nla::Nla, rhs::HVector, expected_density::Float64)
    apply_basis_matrix_col_scale!(nla, rhs)
    btran_in_scaled_space!(nla, rhs, expected_density)
    apply_basis_matrix_row_scale!(nla, rhs)
    return rhs
end

"""`HSimplexNla::rowEp2NormInScaledSpace`."""
function row_ep_2norm_in_scaled_space(nla::Nla, iRow::Int, row_ep::HVector)
    nla.scale === nothing && return norm2(row_ep)
    row_scale = nla.scale.row
    col_scale_value = basic_col_scale_factor(nla, iRow)
    total = 0.0
    use_indices, to_entry = sparse_loop_style(row_ep.count, nla.num_row)
    for iEntry ∈ 1:to_entry
        r = use_indices ? row_ep.index[iEntry] : iEntry
        value = row_ep.array[r] / (row_scale[r] * col_scale_value)
        total += value * value
    end
    return total
end

"""`HSimplexNla::transformForUpdate` — scale `aq`/`ep` to factor units."""
function transform_for_update!(nla::Nla, aq::HVector, ep::HVector,
    variable_in::Int, row_out::Int)
    nla.scale === nothing && return nothing
    cq_scale_factor = variable_scale_factor(nla, variable_in)
    for ix ∈ 1:aq.packCount
        aq.packValue[ix] *= cq_scale_factor
    end
    aq.array[row_out] *= cq_scale_factor
    cp_scale_factor = basic_col_scale_factor(nla, row_out)
    aq.array[row_out] /= cp_scale_factor
    for ix ∈ 1:ep.packCount
        ep.packValue[ix] /= cp_scale_factor
    end
    return nothing
end

"""`HSimplexNla::update` — delegate to `HFactor` (no ProductFormUpdate)."""
function update!(nla::Nla, aq::HVector, ep::HVector, iRow::Int)
    clear!(nla.factor.refactor_info)
    update!(nla.factor, aq, ep, iRow)
    return nothing
end
