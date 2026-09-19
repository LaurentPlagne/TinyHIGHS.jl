# Port of `highs/util/HighsSparseMatrix.{h,cpp}` — M2 subset:
# columnwise/rowwise structure, matrix products, `computeDot`/`collectAj`,
# and pricing (`priceByColumn`, `priceByRow`, `priceByRowWithSwitch`) in double
# and compensated quad precision (`HighsCDouble`, `HighsSparseVectorSum`).
#
# Column/row additions and deletions are omitted.
#
# 1-based indexing: `start` contains element positions (1 = first element),
# `index` contains 1-based indices; `format` ∈ (kColwise, kRowwise,
# kRowwisePartitioned).

const kColwise = 1
const kRowwise = 2
const kRowwisePartitioned = 3

"""
    SparseVectorSum()

`HighsSparseVectorSum`: compensated accumulation (`HighsCDouble`) of a sparse
vector during quad-precision rowwise pricing. `values` is dense and `nonzeroinds`
holds the touched positions in order of insertion (compacted by `cleanup!` via swaps).

The buffer belongs to the owning `SparseMatrix` (`quad_sum`) and is reset on each
call to quad pricing without reallocating, unlike upstream's local variable.
"""
mutable struct SparseVectorSum
    values::Vector{CDouble}
    nonzeroinds::Vector{Int}
end

SparseVectorSum() = SparseVectorSum(CDouble[], Int[])

"""`HighsSparseVectorSum::setDimension` reset (upstream creates a fresh object;
here the buffer is reused). Zeroing is unconditional: dense pricing writes
positions not present in `nonzeroinds`, so partial clearing would leak into subsequent calls."""
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
`HighsSparseVectorSum::add(index, value)`: `values[index] += value`, or
insert if previously zero. A zero result is replaced by `floatmin(Float64)`
(smallest positive normal) so the position remains marked non-zero; `cleanup!` removes it.
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

"""`HighsSparseVectorSum::cleanup`: drops values where `|x| ≤ kHighsTiny`."""
function cleanup!(s::SparseVectorSum)
    num_nz = length(s.nonzeroinds)
    for i ∈ num_nz:-1:1
        pos = s.nonzeroinds[i]
        if abs(Float64(s.values[pos])) <= kHighsTiny
            s.values[pos] = CDouble(0.0)
            num_nz -= 1
            # Upstream 0-based `numNz` position -> 1-based `num_nz + 1`;
            # when equal to `i`, swap is a no-op (last element).
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

HiGHS sparse matrix, column-wise (CSC 1-based) by default.
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
    (num_col >= 0 && num_row >= 0) || throw(ArgumentError("dimensions must be non-negative"))
    return SparseMatrix(kColwise, num_col, num_row, fill(1, num_col + 1),
        Int[], Int[], Float64[], SparseVectorSum())
end

function SparseMatrix(num_col::Int, num_row::Int, a_start::Vector{Int},
    a_index::Vector{Int}, a_value::Vector{Float64})
    (num_col >= 0 && num_row >= 0) || throw(ArgumentError("dimensions must be non-negative"))
    length(a_start) == num_col + 1 ||
        throw(ArgumentError("a_start must have num_col + 1 entries"))
    length(a_index) == length(a_value) ||
        throw(ArgumentError("a_index and a_value must have identical lengths"))
    (isempty(a_start) || a_start[1] == 1) ||
        throw(ArgumentError("a_start[1] must be 1 (1-based)"))
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

"""`HighsSparseMatrix::range` (absolute values)."""
function range_abs(m::SparseMatrix)
    mn, mx = Inf, 0.0
    for iEl ∈ 1:num_nz(m)
        v = abs(m.value[iEl])
        mn = min(mn, v)
        mx = max(mx, v)
    end
    return mn, mx
end

"""`HighsSparseMatrix::ensureRowwise` — in-place transposition."""
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

"""`HighsSparseMatrix::ensureColwise` — in-place transposition."""
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

`HighsSparseMatrix::createRowwisePartitioned`: rowwise view where entries
of each row are organized into two sections: `[start, p_end)` for columns
in the partition, and `[p_end, start[i+1])` for the others.
`in_partition === nothing` puts all columns into the partition.
"""
function create_rowwise_partitioned!(dst::SparseMatrix, src::SparseMatrix,
    in_partition::Union{Nothing,Vector{Bool}}=nothing)
    is_colwise(src) || throw(ArgumentError("colwise source matrix required"))
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

`HighsSparseMatrix::update`: `var_in` enters the basis (leaves partition),
`var_out` leaves the basis (enters partition). `matrix` is the original
colwise matrix. Logical variables (`> num_col`) have no explicit column entries.
"""
function update!(m::SparseMatrix, var_in::Int, var_out::Int,
    matrix::SparseMatrix)
    m.format == kRowwisePartitioned ||
        throw(ArgumentError("update requires kRowwisePartitioned view"))
    is_colwise(matrix) || throw(ArgumentError("colwise matrix required"))
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

Column/row scaling factors (`HighsScale`, HStruct.h): the scaled matrix
equals `A[i,j] * col[j] * row[i]`. `cost` holds the objective scale factor
(1.0 for simplex strategies), `strategy` the strategy that generated
the factors, and `has_scaling` their validity. Factors are computed as powers of two,
ensuring exact reversibility in floating-point arithmetic.
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

"""Ready-to-use scale (NLA tests: factors provided and valid)."""
Scale(col::Vector{Float64}, row::Vector{Float64}) =
    Scale(col, row, 1.0, true, length(col), length(row),
        kSimplexScaleStrategyOff)

"""Empty `HighsScale` (`HighsLp::clearScale`)."""
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

"""`HighsSparseMatrix::unapplyScale` — divide by `col[j] * row[i]`."""
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
    is_rowwise(m) || throw(ArgumentError("getRow requires rowwise view"))
    return copy(m.index[m.start[iRow]:(m.start[iRow + 1] - 1)]),
    copy(m.value[m.start[iRow]:(m.start[iRow + 1] - 1)])
end

"""`HighsSparseMatrix::getCol` — `(indices, values)`, copy."""
function get_col(m::SparseMatrix, iCol::Int)
    is_colwise(m) || throw(ArgumentError("getCol requires colwise view"))
    return copy(m.index[m.start[iCol]:(m.start[iCol + 1] - 1)]),
    copy(m.value[m.start[iCol]:(m.start[iCol + 1] - 1)])
end

"""`HighsSparseMatrix::product` — `result = A x` (result reset to zero)."""
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

"""`HighsSparseMatrix::alphaProductPlusY` — `y += alpha * A x` (or `A^T x`)."""
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
    is_colwise(m) || throw(ArgumentError("computeDot requires colwise view"))
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
    is_colwise(m) || throw(ArgumentError("collectAj requires colwise view"))
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
`HighsSparseMatrix::priceByColumn` — reduced cost pricing column by column.
`quad_precision` accumulates each dot product in `HighsCDouble`
(upstream specifies this flag as the first argument).
"""
function price_by_column!(m::SparseMatrix, result::HVector,
    column::HVector, quad_precision::Bool=false)
    is_colwise(m) || throw(ArgumentError("priceByColumn requires colwise view"))
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
    is_rowwise(m) || throw(ArgumentError("priceByRowDenseResult requires rowwise view"))
    @inbounds for ix ∈ from_index:column.count
        iRow = column.index[ix]
        multiplier = column.array[iRow]
        # `p_end` is exclusive in upstream (0-based) -> last active
        # position in 1-based is `p_end - 1`.
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

"""`HighsSparseMatrix::priceByRowDenseResult` (quad precision, overloading
`vector<HighsCDouble>&` variant)."""
function price_by_row_dense_result!(result::Vector{CDouble}, m::SparseMatrix,
    column::HVector, from_index::Int)
    is_rowwise(m) || throw(ArgumentError("priceByRowDenseResult requires rowwise view"))
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

`HighsSparseMatrix::priceByRowWithSwitch`: hyper-sparse rowwise pricing,
switching dynamically to dense pricing when density threshold is exceeded.
`from_index` is 1-based. Under quad precision, accumulation uses `m.quad_sum`
and `result.count` is not tracked during the hyper-sparse phase (upstream
only checks density threshold against 0).
"""
function price_by_row_with_switch!(m::SparseMatrix, result::HVector,
    column::HVector, expected_density::Float64, from_index::Int,
    switch_density::Float64, quad_precision::Bool=false)
    is_rowwise(m) || throw(ArgumentError("priceByRowWithSwitch requires rowwise view"))
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
        # Incomplete price: finish in dense, then reconstruct indices.
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
        # Complete price: `nonzeroinds` (compacted by `cleanup!`) contains
        # non-zero indices; values are converted back to Float64.
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

"""`HighsSparseMatrix::priceByRow` — pure hyper-sparse rowwise pricing."""
function price_by_row!(m::SparseMatrix, result::HVector, column::HVector,
    quad_precision::Bool=false)
    return price_by_row_with_switch!(m, result, column, -Inf, 1, Inf,
        quad_precision)
end
