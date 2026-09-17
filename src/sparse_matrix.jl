# Portage de `highs/util/HighsSparseMatrix.{h,cpp}` — sous-ensemble M2 :
# structure colwise/rowwise, produits, `computeDot`/`collectAj` et pricing
# (`priceByColumn`, `priceByRow`, `priceByRowWithSwitch`) en précision double
# **et** quad (`HighsCDouble`, `HighsSparseVectorSum`).
#
# Non porté ici : ajout/suppression de colonnes/lignes.
#
# Convention 1-based : `start` porte des positions (1 = premier élément),
# `index` des indices 1-based ; `format` ∈ (kColwise, kRowwise,
# kRowwisePartitioned).

const kColwise = 1
const kRowwise = 2
const kRowwisePartitioned = 3

"""
    SparseVectorSum()

`HighsSparseVectorSum` : accumulation compensée (`HighsCDouble`) d'un vecteur
creux pendant le pricing rowwise quad. `values` est dense et `nonzeroinds`
porte les positions touchées, dans l'ordre d'insertion (modifié par
`cleanup!`, qui compacte par échange).

Le tampon appartient à la `SparseMatrix` qui l'utilise (`quad_sum`) : il est
réarmé à chaque appel de pricing quad au lieu d'être réalloué, contrairement à
la variable locale de la source.
"""
mutable struct SparseVectorSum
    values::Vector{CDouble}
    nonzeroinds::Vector{Int}
end

SparseVectorSum() = SparseVectorSum(CDouble[], Int[])

"""`HighsSparseVectorSum::setDimension` remis à zéro (la source part d'un
objet neuf ; ici le tampon est réutilisé). Le remplissage est inconditionnel :
la branche dense écrit des positions qui ne sont pas dans `nonzeroinds`, un
zéro ciblé les laisserait fuiter dans l'appel suivant."""
function reset!(s::SparseVectorSum, dim::Int)
    if length(s.values) != dim
        s.values = zeros(CDouble, dim)
        sizehint!(s.nonzeroinds, dim)
    else
        fill!(s.values, CDouble(0.0))
    end
    empty!(s.nonzeroinds)
    return s
end

"""
`HighsSparseVectorSum::add(index, value)` : `values[index] += value`, ou
insertion si la position était nulle. Un résultat nul est remplacé par
`numeric_limits<double>::min()` (plus petit normal positif) pour que la
position reste marquée comme non nulle ; `cleanup!` l'enlèvera.
"""
function add!(s::SparseVectorSum, index::Int, value::Float64)
    if Float64(s.values[index]) != 0.0
        s.values[index] = s.values[index] + value
    else
        s.values[index] = CDouble(value)
        push!(s.nonzeroinds, index)
    end
    if Float64(s.values[index]) == 0.0
        s.values[index] = CDouble(floatmin(Float64))
    end
    return s
end

"""`HighsSparseVectorSum::cleanup` : retire les valeurs `|x| ≤ kHighsTiny`."""
function cleanup!(s::SparseVectorSum)
    num_nz = length(s.nonzeroinds)
    for i ∈ num_nz:-1:1
        pos = s.nonzeroinds[i]
        if abs(Float64(s.values[pos])) <= kHighsTiny
            s.values[pos] = CDouble(0.0)
            num_nz -= 1
            # Position 0-based `numNz` de la source → 1-based `num_nz + 1` ;
            # quand elle vaut `i`, l'échange est un no-op (dernier élément).
            s.nonzeroinds[i], s.nonzeroinds[num_nz + 1] =
                s.nonzeroinds[num_nz + 1], s.nonzeroinds[i]
        end
    end
    resize!(s.nonzeroinds, num_nz)
    return s
end

"""
    SparseMatrix(num_col, num_row, a_start, a_index, a_value)
    SparseMatrix(num_col, num_row)

Matrice creuse de HiGHS, vue colonne par défaut (`a_*` en CSC 1-based).
"""
mutable struct SparseMatrix
    format::Int
    num_col::Int
    num_row::Int
    start::Vector{Int}
    p_end::Vector{Int}
    index::Vector{Int}
    value::Vector{Float64}
    quad_sum::SparseVectorSum
end

function SparseMatrix(num_col::Int, num_row::Int)
    (num_col >= 0 && num_row >= 0) || throw(ArgumentError("dimensions négatives"))
    return SparseMatrix(kColwise, num_col, num_row, fill(1, num_col + 1),
        Int[], Int[], Float64[], SparseVectorSum())
end

function SparseMatrix(num_col::Int, num_row::Int, a_start::Vector{Int},
    a_index::Vector{Int}, a_value::Vector{Float64})
    (num_col >= 0 && num_row >= 0) || throw(ArgumentError("dimensions négatives"))
    length(a_start) == num_col + 1 ||
        throw(ArgumentError("a_start doit avoir num_col+1 entrées"))
    length(a_index) == length(a_value) ||
        throw(ArgumentError("a_index et a_value de tailles différentes"))
    (isempty(a_start) || a_start[1] == 1) ||
        throw(ArgumentError("a_start[1] doit valoir 1 (1-based)"))
    return SparseMatrix(kColwise, num_col, num_row, copy(a_start), Int[],
        copy(a_index), copy(a_value), SparseVectorSum())
end

Base.copy(m::SparseMatrix) = SparseMatrix(m.format, m.num_col, m.num_row,
    copy(m.start), copy(m.p_end), copy(m.index), copy(m.value),
    SparseVectorSum())

is_rowwise(m::SparseMatrix) = m.format == kRowwise ||
                              m.format == kRowwisePartitioned
is_colwise(m::SparseMatrix) = m.format == kColwise

"""`HighsSparseMatrix::numNz`."""
function num_nz(m::SparseMatrix)
    if is_colwise(m)
        return m.start[m.num_col + 1] - 1
    else
        return m.start[m.num_row + 1] - 1
    end
end

"""`HighsSparseMatrix::range` (valeurs absolues)."""
function range_abs(m::SparseMatrix)
    mn, mx = Inf, 0.0
    for iEl ∈ 1:num_nz(m)
        v = abs(m.value[iEl])
        mn = min(mn, v)
        mx = max(mx, v)
    end
    return mn, mx
end

"""`HighsSparseMatrix::ensureRowwise` — transposition en place."""
function ensure_rowwise!(m::SparseMatrix)
    is_rowwise(m) && return m
    num_col, num_row, nnz = m.num_col, m.num_row, num_nz(m)
    if nnz == 0
        m.start = fill(1, num_row + 1)
        empty!(m.index)
        empty!(m.value)
    else
        Astart, Aindex, Avalue = copy(m.start), copy(m.index), copy(m.value)
        start = Vector{Int}(undef, num_row + 1)
        row_length = zeros(Int, num_row)
        for iEl ∈ 1:nnz
            row_length[Aindex[iEl]] += 1
        end
        start[1] = 1
        for iRow ∈ 1:num_row
            start[iRow + 1] = start[iRow] + row_length[iRow]
        end
        index = Vector{Int}(undef, nnz)
        value = Vector{Float64}(undef, nnz)
        ends = copy(start[1:num_row])
        for iCol ∈ 1:num_col
            for iEl ∈ Astart[iCol]:(Astart[iCol + 1] - 1)
                iRow = Aindex[iEl]
                i_to = ends[iRow]
                ends[iRow] += 1
                index[i_to] = iCol
                value[i_to] = Avalue[iEl]
            end
        end
        m.start, m.index, m.value = start, index, value
    end
    m.format = kRowwise
    return m
end

"""`HighsSparseMatrix::ensureColwise` — transposition en place."""
function ensure_colwise!(m::SparseMatrix)
    is_colwise(m) && return m
    num_col, num_row, nnz = m.num_col, m.num_row, num_nz(m)
    if nnz == 0
        m.start = fill(1, num_col + 1)
        empty!(m.index)
        empty!(m.value)
    else
        ARstart, ARindex, ARvalue = copy(m.start), copy(m.index), copy(m.value)
        start = Vector{Int}(undef, num_col + 1)
        col_length = zeros(Int, num_col)
        for iEl ∈ 1:nnz
            col_length[ARindex[iEl]] += 1
        end
        start[1] = 1
        for iCol ∈ 1:num_col
            start[iCol + 1] = start[iCol] + col_length[iCol]
        end
        index = Vector{Int}(undef, nnz)
        value = Vector{Float64}(undef, nnz)
        ends = copy(start[1:num_col])
        for iRow ∈ 1:num_row
            for iEl ∈ ARstart[iRow]:(ARstart[iRow + 1] - 1)
                iCol = ARindex[iEl]
                i_to = ends[iCol]
                ends[iCol] += 1
                index[i_to] = iRow
                value[i_to] = ARvalue[iEl]
            end
        end
        m.start, m.index, m.value = start, index, value
    end
    m.format = kColwise
    return m
end

"""`HighsSparseMatrix::setFormat`."""
function set_format!(m::SparseMatrix, desired::Int)
    desired == kColwise ? ensure_colwise!(m) : ensure_rowwise!(m)
    return m
end

"""
    create_rowwise_partitioned!(dst, src, in_partition)

`HighsSparseMatrix::createRowwisePartitioned` : vue rowwise où les entrées de
chaque ligne sont rangées en deux sections, `[start, p_end)` pour les colonnes
dans la partition et `[p_end, start[i+1])` pour les autres.
`in_partition === nothing` place toutes les colonnes dans la partition.
"""
function create_rowwise_partitioned!(dst::SparseMatrix, src::SparseMatrix,
    in_partition::Union{Nothing,Vector{Bool}}=nothing)
    is_colwise(src) || throw(ArgumentError("source colwise requise"))
    all_in = in_partition === nothing
    num_col, num_row, nnz = src.num_col, src.num_row, num_nz(src)
    start = Vector{Int}(undef, num_row + 1)
    p_end_pos = Vector{Int}(undef, num_row)
    tail_pos = Vector{Int}(undef, num_row)
    part_count = zeros(Int, num_row)
    tail_count = zeros(Int, num_row)
    for iCol ∈ 1:num_col
        in_part = all_in || in_partition[iCol]
        for iEl ∈ src.start[iCol]:(src.start[iCol + 1] - 1)
            iRow = src.index[iEl]
            in_part ? (part_count[iRow] += 1) : (tail_count[iRow] += 1)
        end
    end
    start[1] = 1
    for iRow ∈ 1:num_row
        start[iRow + 1] = start[iRow] + part_count[iRow] + tail_count[iRow]
        p_end_pos[iRow] = start[iRow]
        tail_pos[iRow] = start[iRow] + part_count[iRow]
    end
    index = Vector{Int}(undef, nnz)
    value = Vector{Float64}(undef, nnz)
    for iCol ∈ 1:num_col
        in_part = all_in || in_partition[iCol]
        for iEl ∈ src.start[iCol]:(src.start[iCol + 1] - 1)
            iRow = src.index[iEl]
            if in_part
                pos = p_end_pos[iRow]
                p_end_pos[iRow] += 1
            else
                pos = tail_pos[iRow]
                tail_pos[iRow] += 1
            end
            index[pos] = iCol
            value[pos] = src.value[iEl]
        end
    end
    dst.format = kRowwisePartitioned
    dst.num_col = num_col
    dst.num_row = num_row
    dst.start = start
    dst.p_end = p_end_pos
    dst.index = index
    dst.value = value
    return dst
end

"""
    update!(m, var_in, var_out, matrix)

`HighsSparseMatrix::update` : `var_in` entre en base (sort de la partition),
`var_out` en sort (entre dans la partition). `matrix` est la matrice colwise
d'origine. Les variables logiques (`> num_col`) n'ont pas d'entrées.
"""
function update!(m::SparseMatrix, var_in::Int, var_out::Int,
    matrix::SparseMatrix)
    m.format == kRowwisePartitioned ||
        throw(ArgumentError("update exige la vue kRowwisePartitioned"))
    is_colwise(matrix) || throw(ArgumentError("matrice colwise requise"))
    if var_in <= m.num_col
        for iEl ∈ matrix.start[var_in]:(matrix.start[var_in + 1] - 1)
            iRow = matrix.index[iEl]
            m.p_end[iRow] -= 1
            i_swap = m.p_end[iRow]
            i_find = m.start[iRow]
            while m.index[i_find] != var_in
                i_find += 1
            end
            m.index[i_find], m.index[i_swap] = m.index[i_swap], m.index[i_find]
            m.value[i_find], m.value[i_swap] = m.value[i_swap], m.value[i_find]
        end
    end
    if var_out <= m.num_col
        for iEl ∈ matrix.start[var_out]:(matrix.start[var_out + 1] - 1)
            iRow = matrix.index[iEl]
            i_find = m.p_end[iRow]
            i_swap = m.p_end[iRow]
            m.p_end[iRow] += 1
            while m.index[i_find] != var_out
                i_find += 1
            end
            m.index[i_find], m.index[i_swap] = m.index[i_swap], m.index[i_find]
            m.value[i_find], m.value[i_swap] = m.value[i_swap], m.value[i_find]
        end
    end
    return m
end

"""
    Scale(col, row)

Facteurs d'échelle colonne/ligne (`HighsScale`, HStruct.h) : la matrice
échelonnée vaut `A[i,j] * col[j] * row[i]`. `cost` porte l'échelle d'objectif
(1.0 pour les stratégies du simplexe), `strategy` la stratégie qui a produit
les facteurs et `has_scaling` leur applicabilité. Les facteurs calculés sont
des puissances de deux, donc exactement inversibles.
"""
struct Scale
    col::Vector{Float64}
    row::Vector{Float64}
    cost::Float64
    has_scaling::Bool
    num_col::Int
    num_row::Int
    strategy::Int
end

"""Scale prêt à l'emploi (tests NLA : facteurs fournis, applicables)."""
Scale(col::Vector{Float64}, row::Vector{Float64}) =
    Scale(col, row, 1.0, true, length(col), length(row),
        kSimplexScaleStrategyOff)

"""`HighsScale` vide (`HighsLp::clearScale`)."""
Scale() = Scale(Float64[], Float64[], 1.0, false, 0, 0,
    kSimplexScaleStrategyOff)

"""`HighsSparseMatrix::scaleCol`."""
function scale_col!(m::SparseMatrix, col::Int, col_scale::Float64)
    if is_colwise(m)
        for iEl ∈ m.start[col]:(m.start[col + 1] - 1)
            m.value[iEl] *= col_scale
        end
    else
        for iRow ∈ 1:m.num_row
            for iEl ∈ m.start[iRow]:(m.start[iRow + 1] - 1)
                m.index[iEl] == col && (m.value[iEl] *= col_scale)
            end
        end
    end
    return m
end

"""`HighsSparseMatrix::scaleRow`."""
function scale_row!(m::SparseMatrix, row::Int, row_scale::Float64)
    if is_colwise(m)
        for iCol ∈ 1:m.num_col
            for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                m.index[iEl] == row && (m.value[iEl] *= row_scale)
            end
        end
    else
        for iEl ∈ m.start[row]:(m.start[row + 1] - 1)
            m.value[iEl] *= row_scale
        end
    end
    return m
end

"""`HighsSparseMatrix::applyScale` — `A[i,j] *= col[j] * row[i]`."""
function apply_scale!(m::SparseMatrix, scale::Scale)
    if is_colwise(m)
        for iCol ∈ 1:m.num_col
            for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                m.value[iEl] *= scale.col[iCol] * scale.row[m.index[iEl]]
            end
        end
    else
        for iRow ∈ 1:m.num_row
            for iEl ∈ m.start[iRow]:(m.start[iRow + 1] - 1)
                m.value[iEl] *= scale.col[m.index[iEl]] * scale.row[iRow]
            end
        end
    end
    return m
end

"""`HighsSparseMatrix::applyColScale`."""
function apply_col_scale!(m::SparseMatrix, scale::Scale)
    if is_colwise(m)
        for iCol ∈ 1:m.num_col
            for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                m.value[iEl] *= scale.col[iCol]
            end
        end
    else
        for iRow ∈ 1:m.num_row
            for iEl ∈ m.start[iRow]:(m.start[iRow + 1] - 1)
                m.value[iEl] *= scale.col[m.index[iEl]]
            end
        end
    end
    return m
end

"""`HighsSparseMatrix::applyRowScale`."""
function apply_row_scale!(m::SparseMatrix, scale::Scale)
    if is_colwise(m)
        for iCol ∈ 1:m.num_col
            for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                m.value[iEl] *= scale.row[m.index[iEl]]
            end
        end
    else
        for iRow ∈ 1:m.num_row
            for iEl ∈ m.start[iRow]:(m.start[iRow + 1] - 1)
                m.value[iEl] *= scale.row[iRow]
            end
        end
    end
    return m
end

"""`HighsSparseMatrix::unapplyScale` — division par `col[j] * row[i]`."""
function unapply_scale!(m::SparseMatrix, scale::Scale)
    if is_colwise(m)
        for iCol ∈ 1:m.num_col
            for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                m.value[iEl] /= scale.col[iCol] * scale.row[m.index[iEl]]
            end
        end
    else
        for iRow ∈ 1:m.num_row
            for iEl ∈ m.start[iRow]:(m.start[iRow + 1] - 1)
                m.value[iEl] /= scale.col[m.index[iEl]] * scale.row[iRow]
            end
        end
    end
    return m
end

"""`HighsSparseMatrix::getRow` — `(indices, valeurs)`, copie."""
function get_row(m::SparseMatrix, iRow::Int)
    is_rowwise(m) || throw(ArgumentError("getRow exige la vue rowwise"))
    return copy(m.index[m.start[iRow]:(m.start[iRow + 1] - 1)]),
    copy(m.value[m.start[iRow]:(m.start[iRow + 1] - 1)])
end

"""`HighsSparseMatrix::getCol` — `(indices, valeurs)`, copie."""
function get_col(m::SparseMatrix, iCol::Int)
    is_colwise(m) || throw(ArgumentError("getCol exige la vue colwise"))
    return copy(m.index[m.start[iCol]:(m.start[iCol + 1] - 1)]),
    copy(m.value[m.start[iCol]:(m.start[iCol + 1] - 1)])
end

"""`HighsSparseMatrix::product` — `result = A x` (result remis à zéro)."""
function product!(result::Vector{Float64}, m::SparseMatrix,
    x::Vector{Float64})
    fill!(result, 0.0)
    if is_colwise(m)
        for iCol ∈ 1:m.num_col
            for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                result[m.index[iEl]] += x[iCol] * m.value[iEl]
            end
        end
    else
        for iRow ∈ 1:m.num_row
            for iEl ∈ m.start[iRow]:(m.start[iRow + 1] - 1)
                result[iRow] += x[m.index[iEl]] * m.value[iEl]
            end
        end
    end
    return result
end

"""`HighsSparseMatrix::productTranspose` — `result = A^T x`."""
function product_transpose!(result::Vector{Float64}, m::SparseMatrix,
    x::Vector{Float64})
    fill!(result, 0.0)
    if is_colwise(m)
        for iCol ∈ 1:m.num_col
            for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                result[iCol] += x[m.index[iEl]] * m.value[iEl]
            end
        end
    else
        for iRow ∈ 1:m.num_row
            for iEl ∈ m.start[iRow]:(m.start[iRow + 1] - 1)
                result[m.index[iEl]] += x[iRow] * m.value[iEl]
            end
        end
    end
    return result
end

"""`HighsSparseMatrix::alphaProductPlusY` — `y += alpha * A x` (ou `A^T x`)."""
function alpha_product_plus_y!(y::Vector{Float64}, alpha::Float64,
    m::SparseMatrix, x::Vector{Float64}; transpose::Bool=false)
    if is_colwise(m)
        if transpose
            for iCol ∈ 1:m.num_col
                for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                    y[iCol] += alpha * m.value[iEl] * x[m.index[iEl]]
                end
            end
        else
            for iCol ∈ 1:m.num_col
                for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                    y[m.index[iEl]] += alpha * m.value[iEl] * x[iCol]
                end
            end
        end
    else
        if transpose
            for iRow ∈ 1:m.num_row
                for iEl ∈ m.start[iRow]:(m.start[iRow + 1] - 1)
                    y[m.index[iEl]] += alpha * m.value[iEl] * x[iRow]
                end
            end
        else
            for iRow ∈ 1:m.num_row
                for iEl ∈ m.start[iRow]:(m.start[iRow + 1] - 1)
                    y[iRow] += alpha * m.value[iEl] * x[m.index[iEl]]
                end
            end
        end
    end
    return y
end

"""`HighsSparseMatrix::computeDot` — `a_use_col · array`."""
function compute_dot(m::SparseMatrix, array::Vector{Float64}, use_col::Int)
    is_colwise(m) || throw(ArgumentError("computeDot exige la vue colwise"))
    result = 0.0
    if use_col <= m.num_col
        for iEl ∈ m.start[use_col]:(m.start[use_col + 1] - 1)
            result += array[m.index[iEl]] * m.value[iEl]
        end
    else
        result = array[use_col - m.num_col]
    end
    return result
end

"""`HighsSparseMatrix::collectAj` — `column += multiplier * a_use_col`."""
function collect_aj!(m::SparseMatrix, column::HVector, use_col::Int,
    multiplier::Float64)
    is_colwise(m) || throw(ArgumentError("collectAj exige la vue colwise"))
    if use_col <= m.num_col
        @inbounds for iEl ∈ m.start[use_col]:(m.start[use_col + 1] - 1)
            iRow = m.index[iEl]
            value0 = column.array[iRow]
            value1 = value0 + multiplier * m.value[iEl]
            if value0 == 0.0
                column.count += 1
                column.index[column.count] = iRow
            end
            column.array[iRow] = abs(value1) < kHighsTiny ? kHighsZero : value1
        end
    else
        iRow = use_col - m.num_col
        @inbounds value0 = column.array[iRow]
        value1 = value0 + multiplier
        @inbounds if value0 == 0.0
            column.count += 1
            column.index[column.count] = iRow
        end
        @inbounds column.array[iRow] = abs(value1) < kHighsTiny ? kHighsZero : value1
    end
    return column
end

"""
`HighsSparseMatrix::priceByColumn` — prix des coûts réduits colonne par
colonne. `quad_precision` accumule chaque produit scalaire en `HighsCDouble`
(la source a le drapeau en premier argument).
"""
function price_by_column!(m::SparseMatrix, result::HVector,
    column::HVector, quad_precision::Bool=false)
    is_colwise(m) || throw(ArgumentError("priceByColumn exige la vue colwise"))
    result.count = 0
    @inbounds for iCol ∈ 1:m.num_col
        value = 0.0
        if quad_precision
            quad_value = CDouble(0.0)
            for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                quad_value += column.array[m.index[iEl]] * m.value[iEl]
            end
            value = Float64(quad_value)
        else
            for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                value += column.array[m.index[iEl]] * m.value[iEl]
            end
        end
        if abs(value) > kHighsTiny
            result.array[iCol] = value
            result.count += 1
            result.index[result.count] = iCol
        end
    end
    return result
end

"""`HighsSparseMatrix::priceByRowDenseResult` (double)."""
function price_by_row_dense_result!(result::Vector{Float64}, m::SparseMatrix,
    column::HVector, from_index::Int)
    is_rowwise(m) || throw(ArgumentError("priceByRowDenseResult exige rowwise"))
    @inbounds for ix ∈ from_index:column.count
        iRow = column.index[ix]
        multiplier = column.array[iRow]
        # `p_end` est une fin exclusive (0-based côté source) → dernière
        # position active = p_end - 1 en 1-based.
        to_iEl = m.format == kRowwisePartitioned ? m.p_end[iRow] - 1 :
                 m.start[iRow + 1] - 1
        for iEl ∈ m.start[iRow]:to_iEl
            iCol = m.index[iEl]
            value0 = result[iCol]
            value1 = value0 + multiplier * m.value[iEl]
            result[iCol] = abs(value1) < kHighsTiny ? kHighsZero : value1
        end
    end
    return result
end

"""`HighsSparseMatrix::priceByRowDenseResult` (quad, surnom de la souche
`vector<HighsCDouble>&`)."""
function price_by_row_dense_result!(result::Vector{CDouble}, m::SparseMatrix,
    column::HVector, from_index::Int)
    is_rowwise(m) || throw(ArgumentError("priceByRowDenseResult exige rowwise"))
    @inbounds for ix ∈ from_index:column.count
        iRow = column.index[ix]
        multiplier = column.array[iRow]
        to_iEl = m.format == kRowwisePartitioned ? m.p_end[iRow] - 1 :
                 m.start[iRow + 1] - 1
        for iEl ∈ m.start[iRow]:to_iEl
            iCol = m.index[iEl]
            value1 = result[iCol] + multiplier * m.value[iEl]
            result[iCol] = abs(Float64(value1)) < kHighsTiny ?
                           CDouble(kHighsZero) : value1
        end
    end
    return result
end

"""
    price_by_row_with_switch!(m, result, column, expected_density, from_index,
                              switch_density, quad_precision)

`HighsSparseMatrix::priceByRowWithSwitch` : prix hyper-sparse rowwise, avec
bascule éventuelle vers le prix dense. `from_index` est 1-based. En quad, les
accumulations passent par `m.quad_sum` et `result.count` n'est pas mis à jour
pendant la phase hyper-sparse (la source ne le fait que pour la densité de
bascule, restée à 0).
"""
function price_by_row_with_switch!(m::SparseMatrix, result::HVector,
    column::HVector, expected_density::Float64, from_index::Int,
    switch_density::Float64, quad_precision::Bool=false)
    is_rowwise(m) || throw(ArgumentError("priceByRowWithSwitch exige rowwise"))
    sum = quad_precision ? reset!(m.quad_sum, m.num_col) : nothing
    next_index = from_index
    if expected_density <= kHyperPriceDensity
        inv_num_col = 1.0 / m.num_col
        @inbounds while next_index <= column.count
            iRow = column.index[next_index]
            to_iEl = m.format == kRowwisePartitioned ? m.p_end[iRow] - 1 :
                     m.start[iRow + 1] - 1
            row_num_nz = to_iEl - m.start[iRow] + 1
            local_density = (1.0 * result.count) * inv_num_col
            switch_to_dense = result.count + row_num_nz >= m.num_col ||
                              local_density > switch_density
            switch_to_dense && break
            multiplier = column.array[iRow]
            if quad_precision
                if multiplier != 0.0
                    for iEl ∈ m.start[iRow]:to_iEl
                        add!(sum, m.index[iEl], multiplier * m.value[iEl])
                    end
                end
            elseif multiplier != 0.0
                for iEl ∈ m.start[iRow]:to_iEl
                    iCol = m.index[iEl]
                    value0 = result.array[iCol]
                    value1 = value0 + multiplier * m.value[iEl]
                    if value0 == 0.0
                        result.count += 1
                        result.index[result.count] = iCol
                    end
                    result.array[iCol] = abs(value1) < kHighsTiny ? kHighsZero :
                                         value1
                end
            end
            next_index += 1
        end
    end
    if quad_precision
        cleanup!(sum)
    end
    if next_index <= column.count
        # Prix incomplet : finir en dense, puis reconstruire les indices.
        if quad_precision
            price_by_row_dense_result!(sum.values, m, column, next_index)
            result.count = 0
            @inbounds for iCol ∈ 1:m.num_col
                value1 = Float64(sum.values[iCol])
                if abs(value1) < kHighsTiny
                    result.array[iCol] = 0.0
                else
                    result.array[iCol] = value1
                    result.count += 1
                    result.index[result.count] = iCol
                end
            end
        else
            price_by_row_dense_result!(result.array, m, column, next_index)
            result.count = 0
            @inbounds for iCol ∈ 1:m.num_col
                value1 = result.array[iCol]
                if abs(value1) < kHighsTiny
                    result.array[iCol] = 0.0
                else
                    result.count += 1
                    result.index[result.count] = iCol
                end
            end
        end
    elseif quad_precision
        # Prix complet : `nonzeroinds` (compacté par `cleanup`) porte les
        # indices ; les valeurs sont reconverties en double.
        result.count = length(sum.nonzeroinds)
        for i ∈ 1:result.count
            iRow = sum.nonzeroinds[i]
            result.index[i] = iRow
            result.array[iRow] = Float64(sum.values[iRow])
        end
    else
        tight!(result)
    end
    return result
end

"""`HighsSparseMatrix::priceByRow` — prix hyper-sparse rowwise pur."""
function price_by_row!(m::SparseMatrix, result::HVector, column::HVector,
    quad_precision::Bool=false)
    return price_by_row_with_switch!(m, result, column, -Inf, 1, Inf,
        quad_precision)
end
