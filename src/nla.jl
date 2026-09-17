# Portage partiel de `simplex/HSimplexNla.{h,cpp}` (M2b) : solves en espace
# échelonné et application des échelles autour de `HFactor`.
#
# Non porté : `ProductFormUpdate` (mises à jour PF), `SimplexIterate`,
# `putInvert`/`getInvert`, `addCols`/`addRows`, télémétrie/rapports.
# `consider*Scaling` (calcul des facteurs) reste en M5 ; ici les facteurs sont
# fournis (ou `nothing` = pas d'échelle).
#
# Convention 1-based : `basic_index[i]` est le numéro de variable (1-based) de
# la i-ème colonne de base ; les facteurs d'échelle suivent la même convention
# que `HighsScale` (variable ≤ num_col → `col`, sinon logique → `row`).

const kDensityForIndexing = 0.4

"""
    Nla(factor, num_col, num_row, basic_index; scale=nothing)

Interface de `HFactor` avec échelles NLA (`HSimplexNla`).
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
Contrairement à `HFactor`, la NLA **partage** `factor.basic_index` (en C++, la
classe stocke le pointeur de l'appelant) : `build!` permute le tableau en place
et les facteurs d'échelle restent alignés sur l'ordre des colonnes de base.
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

"""`HSimplexNla::basicColScaleFactor` — `iCol` indexe la base (1-based)."""
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

"""`HSimplexNla::invert` — reconstruction du facteur, tick conservé."""
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

"""`HSimplexNla::ftran` — FTRAN dans l'espace original (échelles appliquées)."""
function ftran!(nla::Nla, rhs::HVector, expected_density::Float64)
    apply_basis_matrix_row_scale!(nla, rhs)
    ftran_in_scaled_space!(nla, rhs, expected_density)
    apply_basis_matrix_col_scale!(nla, rhs)
    return rhs
end

"""`HSimplexNla::btran` — BTRAN dans l'espace original."""
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

"""`HSimplexNla::transformForUpdate` — met `aq`/`ep` à l'échelle du facteur."""
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

"""`HSimplexNla::update` — délègue à `HFactor` (pas de ProductFormUpdate)."""
function update!(nla::Nla, aq::HVector, ep::HVector, iRow::Int)
    clear!(nla.factor.refactor_info)
    update!(nla.factor, aq, ep, iRow)
    return nothing
end
