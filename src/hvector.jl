# Portage de `highs/util/HVectorBase.{h,cpp}` (licence MIT, HiGHS).
import Base: copy!
#
# Convention : tous les indices sont **1-based** (`array[i]`), contrairement à
# la source qui est 0-based ; `count < 0` garde le sens « liste d'indices
# inconnue » (ce n'est pas un index). `index[1:count]` porte les positions des
# valeurs potentiellement non nulles, sans garantie d'ordre ni de non-nullité
# (le C++ ne suit pas les annulations, cf. `saxpy!`).

"""
    HVector(size::Int)

Sparse/dense hybrid vector structure equivalent to HiGHS's `HVectorBase<double>`.

Combines a dense values array `array` of length `size` with an optional non-zero index list
`index[1:count]`. This dual representation allows:
- \$O(1)\$ random access to values via `array[i]`.
- \$O(\\text{nnz})\$ sparse traversal, scatter, and gather operations when `count >= 0`.
- Automatic fallback to dense vector operations when sparsity density exceeds ~30%.

# Fields
- `size::Int`: Dimension of the vector.
- `count::Int`: Number of non-zero entries in `index`. If negative (`< 0`), the index list is invalid/unknown and `array` must be treated as dense.
- `index::Vector{Int}`: 1-based indices of non-zero elements.
- `array::Vector{Float64}`: Dense values array.
- `packFlag::Bool`: Indicates whether packed representation is active.
- `packCount::Int`: Number of packed elements.
- `packIndex` / `packValue`: Packed non-zero indices and values.
- `cwork` / `iwork`: Preallocated scratch buffers for hyper-sparse graph searches and operations.
"""
mutable struct HVector
    size::Int
    count::Int
    index::Vector{Int}
    array::Vector{Float64}
    synthetic_tick::Float64
    packFlag::Bool
    packCount::Int
    packIndex::Vector{Int}
    packValue::Vector{Float64}
    cwork::Vector{UInt8}
    iwork::Vector{Int}
end

function HVector(size::Int)
    size >= 0 || throw(ArgumentError("dimension must be non-negative: $size"))
    return HVector(size, 0, zeros(Int, size), zeros(size), 0.0, false, 0,
        zeros(Int, size), zeros(size), zeros(UInt8, size + 6400),
        zeros(Int, 4 * size))
end

"""
    setup!(v::HVector, size::Int)

Resize and zero-initialize an existing `HVector` to the specified dimension without allocating
new container instances if the capacity is sufficient.
"""
function setup!(v::HVector, size_::Int)
    size_ >= 0 || throw(ArgumentError("dimension must be non-negative: $size_"))
    v.size = size_
    v.count = 0
    resize!(v.index, size_)
    fill!(v.index, 0)
    resize!(v.array, size_)
    fill!(v.array, 0.0)
    resize!(v.cwork, size_ + 6400)
    fill!(v.cwork, 0x00)
    resize!(v.iwork, 4 * size_)
    fill!(v.iwork, 0)
    v.packCount = 0
    resize!(v.packIndex, size_)
    fill!(v.packIndex, 0)
    resize!(v.packValue, size_)
    fill!(v.packValue, 0.0)
    v.packFlag = false
    v.synthetic_tick = 0.0
    return v
end

"""
    reset!(v::HVector)

Reset all values, index lists, pack arrays, and internal work buffers to zero.
"""
function reset!(v::HVector)
    v.count = 0
    v.packCount = 0
    v.packFlag = false
    v.synthetic_tick = 0.0
    fill!(v.index, 0)
    fill!(v.array, 0.0)
    fill!(v.packIndex, 0)
    fill!(v.packValue, 0.0)
    fill!(v.cwork, 0x00)
    fill!(v.iwork, 0)
    return v
end

"""
    clearScalars!(v::HVector)

Reset vector scalars (`packFlag`, `count`, and `synthetic_tick`) to zero without touching data arrays.
"""
function clearScalars!(v::HVector)
    v.packFlag = false
    v.count = 0
    v.synthetic_tick = 0.0
    return v
end

"""
    clear!(v::HVector)

Clear the vector contents to zero.

If non-zero positions are known (`count >= 0`) and sparse (`count <= 0.3 * size`), only
the indexed positions in `array` are zeroed in \$O(\\text{count})\$ time. Otherwise, the full
dense array is filled in \$O(\\text{size})\$.
"""
function clear!(v::HVector)
    if v.count < 0 || v.count > v.size * 0.3
        fill!(v.array, 0.0)
    else
        @inbounds for i ∈ 1:v.count
            idx = v.index[i]
            idx > 0 && (v.array[idx] = 0.0)
        end
    end
    return clearScalars!(v)
end

"""
    tight!(v::HVector)

Zero out numerical noise elements where `abs(value) < kHighsTiny` (1e-13), updating the sparse
index list `index[1:count]` accordingly.
"""
function tight!(v::HVector)
    if v.count < 0
        @inbounds for i ∈ eachindex(v.array)
            abs(v.array[i]) < kHighsTiny && (v.array[i] = 0.0)
        end
        return v
    end
    totalCount = 0
    @inbounds for i ∈ 1:v.count
        my_index = v.index[i]
        value = v.array[my_index]
        if abs(value) >= kHighsTiny
            totalCount += 1
            v.index[totalCount] = my_index
        else
            v.array[my_index] = 0.0
        end
    end
    v.count = totalCount
    return v
end

"""
    pack!(v::HVector)

Pack sparse values and indices into `v.packIndex` and `v.packValue` if `v.packFlag` is set.
"""
function pack!(v::HVector)
    v.packFlag || return v
    v.packFlag = false
    v.packCount = 0
    @inbounds for i ∈ 1:v.count
        ipack = v.index[i]
        v.packCount += 1
        v.packIndex[v.packCount] = ipack
        v.packValue[v.packCount] = v.array[ipack]
    end
    return v
end

"""
    reIndex!(v::HVector)

Reconstruct the non-zero index list `v.index[1:v.count]` by scanning `v.array`.
Skips scanning if the current index list is already valid and sufficiently sparse.
"""
function reIndex!(v::HVector)
    v.count >= 0 && v.count <= v.size * 0.1 && return v
    v.count = 0
    @inbounds for i ∈ eachindex(v.array)
        if v.array[i] != 0.0
            v.count += 1
            v.index[v.count] = i
        end
    end
    return v
end

"""
    copy!(v::HVector, from::HVector)

Copy the sparse representation from `from` into `v`, overwriting existing contents.
"""
function copy!(v::HVector, from::HVector)
    clear!(v)
    v.synthetic_tick = from.synthetic_tick
    count = from.count
    v.count = count
    @inbounds for i ∈ 1:count
        iFrom = from.index[i]
        v.index[i] = iFrom
        v.array[iFrom] = from.array[iFrom]
    end
    return v
end

"""
    norm2(v::HVector) -> Float64

Compute the squared Euclidean 2-norm \$\\sum x_i^2\$ over known non-zero elements in \$O(\\text{count})\$ time.
"""
function norm2(v::HVector)
    result = 0.0
    count = v.count
    array = v.array
    index = v.index
    @inbounds for i ∈ 1:count
        value = array[index[i]]
        result += value * value
    end
    return result
end

"""
    saxpy!(v::HVector, pivotX::Float64, pivot::HVector)

Perform the sparse linear combination `v += pivotX * pivot`.

Maintains the sparse index list `v.index[1:v.count]` without tracking numerical cancellations.
Results smaller than `kHighsTiny` (1e-13) are flushed to `kHighsZero` (0.0).
`v` and `pivot` must be distinct vectors.
"""
function saxpy!(v::HVector, pivotX::Float64, pivot::HVector)
    v === pivot && throw(ArgumentError("saxpy! requires two distinct vectors"))
    workCount = v.count
    for k ∈ 1:pivot.count
        iRow = pivot.index[k]
        x0 = v.array[iRow]
        x1 = x0 + pivotX * pivot.array[iRow]
        if x0 == 0.0
            workCount += 1
            v.index[workCount] = iRow
        end
        v.array[iRow] = abs(x1) < kHighsTiny ? kHighsZero : x1
    end
    v.count = workCount
    return v
end

