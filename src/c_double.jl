# Port of `util/HighsCDouble.h` (MIT License, HiGHS): compensated quad precision
# using two doubles (`hi`, `lo`).
#
# Supported subset: construction from double, conversion to double, `+`, `-`,
# products, and comparisons — as required by `proof_of_primal_infeasibility!`
# and quad-precision pricing in `improveChooseColumnRow` (`src/sparse_matrix.jl`).
# Division (`/=`) and rounding (`floor`/`ceil`) are omitted as unused.
#
# The struct is immutable (matching the C++ value type): `Vector{CDouble}` is
# therefore stored inline without per-element heap allocation, enabling zero-allocation
# accumulator buffers in `SparseVectorSum`.
#
# Algorithms (`two_sum`, `split`, `two_product`) follow Rump, "High precision
# evaluation of nonlinear functions" (2005), transcribed directly from upstream,
# including argument order (upon which error compensation terms depend).

"""
    CDouble(value)

`HighsCDouble`: value and error compensation term, `hi + lo ≈ value`.
"""
struct CDouble
    hi::Float64
    lo::Float64
end

CDouble(val::Float64) = CDouble(val, 0.0)

Base.Float64(x::CDouble) = x.hi + x.lo

Base.zero(::Type{CDouble}) = CDouble(0.0)

"""`HighsCDouble::two_sum` — exact `x + y = a + b`, with `x = fl(a + b)`."""
function two_sum(a::Float64, b::Float64)
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)
    return x, y
end

"""`HighsCDouble::split` — splits 53 bits into two 26-bit parts (`x + y = a`)."""
function split(a::Float64)
    factor = Float64((1 << 27) + 1)
    c = factor * a
    x = c - (c - a)
    y = a - x
    return x, y
end

"""`HighsCDouble::two_product` — exact `x + y = a * b`."""
function two_product(a::Float64, b::Float64)
    x = a * b
    a1, a2 = split(a)
    b1, b2 = split(b)
    y = a2 * b2 - (((x - a1 * b1) - a2 * b1) - a1 * b2)
    return x, y
end

function Base.:+(x::CDouble, v::Float64)
    hi, c = two_sum(v, x.hi)
    return CDouble(hi, x.lo + c)
end

function Base.:+(x::CDouble, v::CDouble)
    res = x + v.hi
    return CDouble(res.hi, res.lo + v.lo)
end

Base.:+(v::Float64, x::CDouble) = x + v

function Base.:-(x::CDouble, v::Float64)
    hi, c = two_sum(x.hi, -v)
    return CDouble(hi, c + x.lo)
end

function Base.:-(x::CDouble, v::CDouble)
    res = x - v.hi
    return CDouble(res.hi, res.lo - v.lo)
end

Base.:-(v::Float64, x::CDouble) = -x + v
Base.:-(x::CDouble) = CDouble(-x.hi, -x.lo)

function Base.:*(x::CDouble, v::Float64)
    hi, lo = two_product(x.hi, v)
    # `res += lo * v` in upstream: renormalized via `operator+=(double)`,
    # not simple accumulation into `lo`.
    return CDouble(hi, lo) + x.lo * v
end

function Base.:*(x::CDouble, v::CDouble)
    res = x * v.hi
    return res + x.hi * v.lo
end

Base.:*(v::Float64, x::CDouble) = x * v

# Upstream compares values (`double(*this) == double(other)`), not raw fields:
# distinct representations of the same real number compare equal.
Base.:(==)(x::CDouble, y::CDouble) = Float64(x) == Float64(y)
Base.:(==)(x::CDouble, v::Float64) = Float64(x) == v
Base.:(==)(v::Float64, x::CDouble) = v == Float64(x)

Base.isless(x::CDouble, v::Float64) = Float64(x) < v
Base.isless(v::Float64, x::CDouble) = v < Float64(x)
Base.isless(x::CDouble, y::CDouble) = Float64(x) < Float64(y)
