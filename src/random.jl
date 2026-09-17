# Portage de `util/HighsRandom.{h,cpp}` (licence MIT, HiGHS) : RNG à état unique
# (xorshift + `pair_hash`). L'état est **consommé de façon déterministe** par le
# simplexe (`initialiseSimplexLpRandomVectors`, puis `chooseNormal` tire l'ordre
# de balayage des rangées, `correctDualInfeasibilities` les shifts) : toute
# divergence d'état change le chemin et les compteurs d'itérations.
#
# `pair_hash<k>(a, b) = (a + c[2k]) * (b + c[2k+1])` en arithmétique 64 bits
# enveloppante (l'énumération des indices est recopiée telle quelle, y compris
# l'absence de `k = 8`).

const kHighsHashConstants = UInt64[
    0xc8497d2a400d9551, 0x80c8963be3e4c2f3, 0x042d8680e260ae5b,
    0x8a183895eeac1536, 0xa94e9c75f80ad6de, 0x7e92251dec62835e,
    0x07294165cb671455, 0x89b0f6212b0a4292, 0x31900011b96bf554,
    0xa44540f8eee2094f, 0xce7ffd372e4c64fc, 0x51c9d471bfe6a10f,
    0x758c2a674483826f, 0xf91a20abe63f8b02, 0xc2a069024a1fcc6f,
    0xd5bb18b70c5dbd59, 0xd510adac6d1ae289, 0x571d069b23050a79,
    0x60873b8872933e06, 0x780481cc19670350, 0x7a48551760216885,
    0xb5d68b918231e6ca, 0xa7e5571699aa5274, 0x7b6d309b2cfdcf01,
    0x04e77c3d474daeff, 0x4dbf099fd7247031, 0x5d70dca901130beb,
    0x9f8b5f0df4182499, 0x293a74c9686092da, 0xd09bdab6840f52b3,
    0xc05d47f3ab302263, 0x6b79e62b884b65d6, 0xa581106fc980c34d,
    0xf081b7145ea2293e, 0xfb27243dd7c3f5ad, 0x5211bf8860ea667f,
    0x9455e65cb2385e7f, 0x0dfaf6731b449b33, 0x4ec98b3c6f5e68c7,
    0x007bfd4a42ae936b, 0x65c93061f8674518, 0x640816f17127c5d1,
    0x6dd4bab17b7c3a74, 0x34d9268c256fa1ba, 0x0b4d0c6b5b50d7f4,
    0x30aa965bc9fadaff, 0xc0ac1d0c2771404d, 0xc5e64509abb76ef2,
    0xd606b11990624a36, 0x0d3f05d242ce2fb7, 0x469a803cb276fe32,
    0xa4a44d177a3e23f4, 0xb9d9a120dcc1ca03, 0x2e15af8165234a2e,
    0x10609ba2720573d4, 0xaa4191b60368d1d5, 0x333dd2300bc57762,
    0xdf6ec48f79fb402f, 0x5ed20fcef1b734fa, 0x4c94924ec8be21ee,
    0x5abe6ad9d131e631, 0xbe10136a522e602d, 0x53671115c340e779,
    0x9f392fe43e2144da]

"""`HighsHashHelpers::pair_hash<k>` (indices de constantes 1-based)."""
pair_hash(k::Int, a::UInt32, b::UInt32) =
    (UInt64(a) + kHighsHashConstants[2k + 1]) *
    (UInt64(b) + kHighsHashConstants[2k + 2])

log2i(n::UInt64) = 63 - leading_zeros(n)

"""Séquence des `k` essayés par `drawUniform` (8 est absent, comme la source)."""
const kDrawUniformIndices = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)

"""
    HighsRandom(seed = 0)

Générateur de HiGHS (`HighsRandom`) : état 64 bits, xorshift + hachage.
"""
mutable struct HighsRandom
    state::UInt64
    function HighsRandom(seed::Integer=0)
        return initialise!(new(0), seed)
    end
end

"""`HighsRandom::initialise`."""
function initialise!(r::HighsRandom, seed::Integer=0)
    r.state = UInt64(seed)
    while true
        r.state = pair_hash(0, UInt32(r.state), UInt32(r.state >> 32))
        r.state ⊻= pair_hash(1, UInt32(r.state >> 32), UInt32(seed)) >> 32
        r.state == 0 || break
    end
    return r
end

"""`HighsRandom::advance` — xorshift 64 bits."""
function advance!(r::HighsRandom)
    s = r.state
    s ⊻= s >> 12
    s ⊻= s << 25
    s ⊻= s >> 27
    r.state = s
    return r
end

"""`HighsRandom::drawUniform` (32 bits)."""
function draw_uniform(r::HighsRandom, sup::UInt32, nbits::Int)
    while true
        advance!(r)
        lo = UInt32(r.state & 0xffffffff)
        hi = UInt32(r.state >> 32)
        for k ∈ kDrawUniformIndices
            val = UInt32(pair_hash(k, lo, hi) >> (64 - nbits))
            val < sup && return val
        end
    end
end

"""`HighsRandom::integer(sup)` — tirage uniforme dans `[0, sup)`."""
function integer(r::HighsRandom, sup::Int)
    sup <= 1 && return 0
    nbits = log2i(UInt64(sup - 1)) + 1
    # `sup` reste sous 2^31 dans le simplexe (rangées/colonnes) : voie 32 bits.
    nbits <= 32 && return Int(draw_uniform(r, UInt32(sup), nbits))
    error("HighsRandom.integer : tirage 64 bits non porté (sup = $sup)")
end

"""`HighsRandom::fraction` — réel dans `(0, 1)`."""
function fraction(r::HighsRandom)
    advance!(r)
    lo = UInt32(r.state & 0xffffffff)
    hi = UInt32(r.state >> 32)
    output = (pair_hash(0, lo, hi) >> (64 - 52)) ⊻
             (pair_hash(1, lo, hi) >> (64 - 26))
    return Float64(1 + output) * 2.2204460492503125e-16
end

"""`HighsRandom::shuffle` sur un vecteur 1-based."""
function shuffle!(r::HighsRandom, data::Vector{Int})
    for i ∈ length(data):-1:2
        pos = integer(r, i) + 1
        data[pos], data[i] = data[i], data[pos]
    end
    return data
end

# --- `HighsHashHelpers` (détection de cyclage) ------------------------------
#
# Le hachage de base est un polynôme évalué modulo le premier de Mersenne
# `2^61-1` ; les fonctions prennent l'indice **0-based** de la source (le port,
# 1-based, convertit avant l'appel).

const kM61 = UInt64(0x1fffffffffffffff)

"""`HighsHashHelpers::multiply_modM61` (`a·b mod 2^61-1`)."""
function multiply_modM61(a::UInt64, b::UInt64)
    ahi = a >> 32
    bhi = b >> 32
    alo = a & 0xffffffff
    blo = b & 0xffffffff
    term_64 = ahi * bhi
    term_32 = ahi * blo + bhi * alo
    term_0 = alo * blo
    term_0 = (term_0 & kM61) + (term_0 >> 61)
    term_0 += ((term_32 >> 29) + (term_32 << 32)) & kM61
    ab61 = (term_64 << 3) | (term_0 >> 61)
    result = (term_0 & kM61) + ab61
    result >= kM61 && (result -= kM61)
    return result
end

"""`HighsHashHelpers::modexp_M61` (`a^e mod 2^61-1`, `e > 0`)."""
function modexp_M61(a::UInt64, e::UInt64)
    result = a
    while e != 1
        result = multiply_modM61(result, result)
        (e & 1) == 1 && (result = multiply_modM61(result, a))
        e >>= 1
    end
    return result
end

"""`HighsHashHelpers::sparse_combine(hash, index)` — `index` 0-based."""
function sparse_combine(hash::UInt64, index::Int)
    a = kHighsHashConstants[(index & 63) + 1] & kM61
    degree = (UInt64(index) >> 6) + 1
    hash += modexp_M61(a, degree)
    hash = (hash >> 61) + (hash & kM61)
    hash >= kM61 && (hash -= kM61)
    return hash
end

"""`HighsHashHelpers::sparse_inverse_combine(hash, index)` — `index` 0-based."""
function sparse_inverse_combine(hash::UInt64, index::Int)
    a = kHighsHashConstants[(index & 63) + 1] & kM61
    degree = (UInt64(index) >> 6) + 1
    hash += kM61 - modexp_M61(a, degree)
    hash = (hash >> 61) + (hash & kM61)
    hash >= kM61 && (hash -= kM61)
    return hash
end
