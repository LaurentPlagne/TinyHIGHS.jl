# Portage de `highs/util/HVectorBase.{h,cpp}` (licence MIT, HiGHS).
import Base: copy!
#
# Convention : tous les indices sont **1-based** (`array[i]`), contrairement à
# la source qui est 0-based ; `count < 0` garde le sens « liste d'indices
# inconnue » (ce n'est pas un index). `index[1:count]` porte les positions des
# valeurs potentiellement non nulles, sans garantie d'ordre ni de non-nullité
# (le C++ ne suit pas les annulations, cf. `saxpy!`).

"""
    HVector(size)

Vecteur creux de travail : tableau dense `array` (taille `size`) + liste
d'indices `index`, exactement `HVectorBase<double>` de HiGHS.
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
    size >= 0 || throw(ArgumentError("taille négative : $size"))
    return HVector(size, 0, zeros(Int, size), zeros(size), 0.0, false, 0,
        zeros(Int, size), zeros(size), zeros(UInt8, size + 6400),
        zeros(Int, 4 * size))
end

"""Réinitialise un vecteur existant — `HVectorBase::setup`."""
function setup!(v::HVector, size_::Int)
    size_ >= 0 || throw(ArgumentError("taille négative : $size_"))
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

"""Réinitialise intégralement un vecteur (données, indices, pack et espaces de travail)."""
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

"""Compteurs et horloge synthétique — `HVectorBase::clearScalars`."""
function clearScalars!(v::HVector)
    v.packFlag = false
    v.count = 0
    v.synthetic_tick = 0.0
    return v
end

"""
    clear!(v)

Remet le vecteur à zéro. Si la liste d'indices est absente ou couvre plus de
30 % de la taille, tout le tableau est rempli (le vecteur peut être dense) ;
sinon seules les positions listées sont zérotées — ce que la garde écarte,
c'est le remplissage complet d'un vecteur creux.
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
    tight!(v)

Met à zéro les valeurs de `array` strictement plus petites que `kHighsTiny` en
magnitude, en maintenant `index` si elle est valide. `count < 0` (indices
inconnus) force le balayage complet du tableau.
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

"""`HVectorBase::pack` — copie valeurs et indices dans `pack*` si demandé."""
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
    reIndex!(v)

Reconstruit la liste d'indices par balayage complet de `array`. La source
n'exécute ce balayage que si la liste courante est absente (`count < 0`) ou si
le vecteur peut être dense (`count > 0.1 * size`) : la garde écarte les vecteurs
creux dont la liste d'indices est déjà valide (HVectorBase.cpp:130).
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

"""`HVectorBase::copy` — copie valeurs et indices, en écrasant la destination."""
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

"""Carré de la norme 2 sur les seules positions listées — `HVectorBase::norm2`."""
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
    saxpy!(v, pivotX, pivot)

`v += pivotX * pivot`, en maintenant la liste d'indices sans suivre les
annulations (une position déjà listée le reste, même si sa valeur s'annule) et
en ramenant les résultats sous `kHighsTiny` à `kHighsZero` — comme la source.
`v` et `pivot` doivent être distincts.
"""
function saxpy!(v::HVector, pivotX::Float64, pivot::HVector)
    v === pivot && throw(ArgumentError("saxpy! exige deux vecteurs distincts"))
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
