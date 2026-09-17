# Portage de `util/HighsCDouble.h` (licence MIT, HiGHS) : « quad precision »
# par compensation (deux doubles `hi`, `lo`).
#
# Sous-ensemble porté : construction depuis un double, conversion en double,
# `+`, `-`, produits et comparaisons — ce qu'exigent
# `proof_of_primal_infeasibility!` et le pricing quad d'`improveChooseColumnRow`
# (`src/sparse_matrix.jl`). Les quotients (`/=`) et les arrondis (`floor`/`ceil`)
# ne sont pas portés : aucun chemin utilisé ne les appelle.
#
# Le type est **immuable** (la source est un value type) : `Vector{CDouble}` est
# alors un tableau de bits (pas d'objet alloué par élément), ce qui permet le
# tampon de somme `SparseVectorSum` sans allocation.
#
# Les algorithmes (`two_sum`, `split`, `two_product`) suivent Rump, « High
# precision evaluation of nonlinear functions » (2005), transcrits de la
# source — ordre des arguments compris (le terme de compensation en dépend).

"""
    CDouble(value)

`HighsCDouble` : valeur et terme de compensation, `hi + lo ≈ value`.
"""
struct CDouble
    hi::Float64
    lo::Float64
end

CDouble(val::Float64) = CDouble(val, 0.0)

Base.Float64(x::CDouble) = x.hi + x.lo

Base.zero(::Type{CDouble}) = CDouble(0.0)

"""`HighsCDouble::two_sum` — `x + y = a + b` exactement, `x = fl(a + b)`."""
function two_sum(a::Float64, b::Float64)
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)
    return x, y
end

"""`HighsCDouble::split` — 53 bits en deux parties de 26 bits (`x + y = a`)."""
function split(a::Float64)
    factor = Float64((1 << 27) + 1)
    c = factor * a
    x = c - (c - a)
    y = a - x
    return x, y
end

"""`HighsCDouble::two_product` — `x + y = a * b` exactement."""
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
    # `res += lo * v` dans la source : renormalisation par `operator+=(double)`,
    # pas une simple accumulation dans `lo`.
    return CDouble(hi, lo) + x.lo * v
end

function Base.:*(x::CDouble, v::CDouble)
    res = x * v.hi
    return res + x.hi * v.lo
end

Base.:*(v::Float64, x::CDouble) = x * v

# La source compare les valeurs (`double(*this) == double(other)`), pas les
# champs : deux représentations différentes d'un même nombre sont égales.
Base.:(==)(x::CDouble, y::CDouble) = Float64(x) == Float64(y)
Base.:(==)(x::CDouble, v::Float64) = Float64(x) == v
Base.:(==)(v::Float64, x::CDouble) = v == Float64(x)

Base.isless(x::CDouble, v::Float64) = Float64(x) < v
Base.isless(v::Float64, x::CDouble) = v < Float64(x)
Base.isless(x::CDouble, y::CDouble) = Float64(x) < Float64(y)
