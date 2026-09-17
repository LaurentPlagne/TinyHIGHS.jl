# Oracle C++ : `HVectorBase` de HiGHS, source INCHANGÉE, pilotée par
# `oracle/build.sh`. Le port Julia est comparé bit-à-bit (hexadécimal %a) opération
# par opération. Voir docs/architecture/portage-julia-simplexe-highs.md §4.1.
using Printf

const ORACLE_BIN = abspath(joinpath(@__DIR__, "..", "oracle", "build", "hvector_oracle"))
const ORACLE_STAMP = abspath(joinpath(@__DIR__, "..", "oracle", "build", "SOURCE_COMMIT"))
const HIGHS_COMMIT = "04024d701f79feb8e2f18bc3df0dffc04ef05088"

hexof(x::Float64) = @sprintf("%a", x)

# --- protocole (indices 0-based côté oracle ; conversion en un seul endroit) ---

function write_state(io::IO, v::HVector)
    println(io, v.size, " ", v.count)
    println(io, join((v.index[i] - 1 for i ∈ 1:max(v.count, 0)), " "))
    println(io, join((hexof(x) for x ∈ v.array), " "))
    return nothing
end

function write_case(io::IO, v::HVector, op::String; pivotX::Float64=0.0,
    pivot::Union{Nothing,HVector}=nothing, dest::Union{Nothing,HVector}=nothing)
    write_state(io, v)
    if op == "saxpy"
        println(io, "saxpy ", hexof(pivotX))
        write_state(io, pivot)
    elseif op == "copy"
        println(io, "copy")
        write_state(io, dest)
    else
        println(io, op)
    end
    return nothing
end

"""État du port, sérialisé dans l'ordre exact de `print_state` (oracle)."""
function record_tokens(v::HVector)
    out = String[string(v.count)]
    append!(out, string.(v.index[1:max(v.count, 0)] .- 1))
    append!(out, hexof.(v.array))
    push!(out, string(v.packFlag ? 1 : 0), string(v.packCount))
    append!(out, string.(v.packIndex[1:v.packCount] .- 1))
    append!(out, hexof.(v.packValue[1:v.packCount]))
    push!(out, hexof(v.synthetic_tick), hexof(norm2(v)))
    return out
end

function apply_op(v::HVector, op::String; pivotX::Float64=0.0,
    pivot::Union{Nothing,HVector}=nothing, dest::Union{Nothing,HVector}=nothing)
    op == "tight" && return tight!(v)
    op == "clear" && return clear!(v)
    op == "reindex" && return reIndex!(v)
    op == "norm2" && return v
    if op == "pack"
        v.packFlag = true
        return pack!(v)
    end
    op == "saxpy" && return saxpy!(v, pivotX, pivot)
    op == "copy" && return copy!(dest, v)
    error("opération inconnue : $op")
end

function run_oracle(input::String)
    out = IOBuffer()
    err = IOBuffer()
    process = run(pipeline(`$ORACLE_BIN`; stdin=IOBuffer(input), stdout=out,
        stderr=err))
    success(process) ||
        error("oracle : code $(process.exitcode) ; $(String(take!(err)))")
    return String(take!(out))
end

"""Découpe la sortie de l'oracle en jetons, un vecteur par cas (taille connue)."""
function parse_records(text::String, sizes::Vector{Int})
    tokens = split(text)
    records = Vector{Vector{String}}()
    pos = 1
    for size ∈ sizes
        record = String[]
        count = parse(Int, tokens[pos])
        push!(record, tokens[pos])
        pos += 1
        for _ ∈ 1:max(count, 0)
            push!(record, tokens[pos])
            pos += 1
        end
        for _ ∈ 1:size
            push!(record, tokens[pos])
            pos += 1
        end
        push!(record, tokens[pos], tokens[pos + 1])
        packCount = parse(Int, tokens[pos + 1])
        pos += 2
        for _ ∈ 1:max(packCount, 0)
            push!(record, tokens[pos])
            pos += 1
        end
        for _ ∈ 1:max(packCount, 0)
            push!(record, tokens[pos])
            pos += 1
        end
        push!(record, tokens[pos], tokens[pos + 1])
        pos += 2
        push!(records, record)
    end
    return records
end

function compare_records(expected::Vector{String}, got::Vector{String})
    length(expected) == length(got) &&
        return ["position $i : attendu $(expected[i]) ≠ oracle $(got[i])"
                for i ∈ eachindex(expected) if expected[i] != got[i]]
    return ["longueur attendue $(length(expected)) ≠ oracle $(length(got))"]
end

function run_comparison(cases)
    input = IOBuffer()
    println(input, length(cases))
    for c ∈ cases
        write_case(input, c.v, c.op; pivotX=c.pivotX, pivot=c.pivot, dest=c.dest)
    end
    text = run_oracle(String(take!(input)))
    records = parse_records(text, [c.v.size for c ∈ cases])
    expected = map(cases) do c
        result = apply_op(c.v, c.op; pivotX=c.pivotX, pivot=c.pivot, dest=c.dest)
        record_tokens(result)
    end
    return expected, records
end

# --- génération des cas -------------------------------------------------------

const VALUE_POOL = Float64[0.0, -0.0, kHighsTiny, prevfloat(kHighsTiny),
    nextfloat(kHighsTiny), -kHighsTiny, kHighsZero, 1.0, -2.5, 1e-15, 1e-13,
    nextfloat(0.0), -nextfloat(0.0)]

random_value(rng::AbstractRNG) =
    rand(rng) < 0.6 ? rand(rng, VALUE_POOL) : randn(rng) * 10.0^rand(rng, -8:3)

# Contrat de la source (à respecter sous peine de comportement indéfini) :
# indices distincts, toute position listée porte une valeur non nulle, et
# `count >= 0` pour `saxpy`. Les zéros exacts ne sont donc testés qu'aux
# positions non listées — c'est aussi le cas réel de la branche `x0 == 0`.
function make_state(rng::AbstractRNG, n::Int, count::Int)
    v = HVector(n)
    cnt = clamp(count, 0, n)
    if cnt > 0
        positions = shuffle(rng, 1:n)[1:cnt]
        for k ∈ 1:cnt
            i = positions[k]
            x = random_value(rng)
            v.index[k] = i
            v.array[i] = iszero(x) ? 1e-30 : x
        end
    end
    v.count = cnt
    return v
end

function make_const_state(n::Int, count::Int, values::Vector{Float64},
    positions::Vector{Int})
    v = HVector(n)
    for (k, i) ∈ enumerate(positions)
        v.index[k] = i
        v.array[i] = values[k]
    end
    v.count = count
    return v
end

function explicit_cases()
    cases = NamedTuple[]
    add = (v, op; pivotX=0.0, pivot=nothing, dest=nothing) ->
        push!(cases, (; v, op, pivotX, pivot, dest))
    rng = MersenneTwister(20260916)

    # Tailles dégénérées.
    for op ∈ ("tight", "clear", "reindex", "pack", "norm2")
        add(HVector(0), op)
        add(HVector(1), op)
    end

    # Gardes de `clear!` : creux (≤ 30 %), dense (> 30 %), liste inconnue.
    add(make_state(rng, 10, 3), "clear")
    add(make_state(rng, 10, 4), "clear")
    let v = make_state(rng, 10, 2)
        v.count = -1
        add(v, "clear")
    end

    # Gardes de `reIndex!` : 10 % exactement (écarté), juste au-dessus
    # (reconstruit), liste inconnue (reconstruit). Une valeur hors liste le prouve.
    let v = make_state(rng, 10, 1)
        v.array[7] = 3.0
        add(v, "reindex")
    end
    add(make_state(rng, 10, 2), "reindex")
    let v = make_state(rng, 10, 2)
        v.count = -1
        add(v, "reindex")
    end

    # `tight!` : seuil exact et ±1 ULP.
    add(make_const_state(4, 4, [kHighsTiny, prevfloat(kHighsTiny),
        nextfloat(kHighsTiny), -kHighsTiny], [1, 2, 3, 4]), "tight")
    let v = make_state(rng, 5, 3)
        v.count = -1
        add(v, "tight")
    end

    # `saxpy!` : x0 nul (position non listée dans v), x0 non nul (conservé),
    # résultat sous le seuil, annulation exacte.
    add(make_state(rng, 6, 2), "saxpy"; pivotX=1.0, pivot=make_state(rng, 6, 3))
    add(make_const_state(6, 2, [1.0, 2.0], [1, 2]), "saxpy"; pivotX=-2.0,
        pivot=make_const_state(6, 3, [3.0, 4.0, 5.0], [3, 4, 5]))
    add(make_const_state(3, 1, [1e-15], [1]), "saxpy"; pivotX=1.0,
        pivot=make_const_state(3, 1, [1.0], [1]))
    add(make_const_state(3, 1, [1.5e-14], [1]), "saxpy"; pivotX=-1.0,
        pivot=make_const_state(3, 1, [1e-14], [1]))     # 5e-15 < seuil → kHighsZero
    add(make_const_state(3, 1, [kHighsTiny], [1]), "saxpy"; pivotX=-1.0,
        pivot=make_const_state(3, 1, [1.0], [1]))       # annulation → kHighsZero
    for pivotX ∈ (0.0, -0.5, 2.0, 1e13)
        add(make_state(rng, 8, 3), "saxpy"; pivotX=pivotX,
            pivot=make_state(rng, 8, 4))
    end

    # `copy!` : destination creuse et dense.
    add(make_state(rng, 8, 3), "copy"; dest=make_state(rng, 8, 2))
    add(make_state(rng, 8, 3), "copy"; dest=make_state(rng, 8, 5))

    return cases
end

function random_cases(rng::AbstractRNG, n_cases::Int)
    cases = NamedTuple[]
    ops = ["tight", "clear", "reindex", "saxpy", "pack", "norm2", "copy"]
    for _ ∈ 1:n_cases
        size = rand(rng, 0:40)
        op = rand(rng, ops)
        # `saxpy` exige `count ≥ 0` dans la source ; les autres ops acceptent -1.
        count = op == "saxpy" ? rand(rng, 0:size) : rand(rng, -1:size)
        v = make_state(rng, size, count)
        pivotX = rand(rng, VALUE_POOL) * 1e6
        pivot = op == "saxpy" ? make_state(rng, size, rand(rng, 0:size)) : nothing
        dest = op == "copy" ? make_state(rng, size, rand(rng, 0:size)) : nothing
        push!(cases, (; v, op, pivotX, pivot, dest))
    end
    return cases
end

@testset "oracle C++ — HVector (source gelée $HIGHS_COMMIT)" begin
    @test isfile(ORACLE_BIN) ||
          error("oracle absent : lancer julia_simplex/oracle/build.sh")
    stamp = split(read(ORACLE_STAMP, String), '\n')
    @test !isempty(stamp) && stamp[1] == HIGHS_COMMIT
    @test length(stamp) > 1 && startswith(stamp[2], "v1.15.1")

    if isfile(ORACLE_BIN)
        @testset "étalonnage : zéro écart contre la source inchangée" begin
            cases = vcat(explicit_cases(), random_cases(MersenneTwister(42), 150))
            expected, records = run_comparison(cases)
            mismatches = String[]
            for i ∈ eachindex(cases)
                append!(mismatches,
                    ("cas $i ($(cases[i].op)) : " * m for m ∈
                     compare_records(expected[i], records[i])))
            end
            @test isempty(mismatches)
            isempty(mismatches) || @info "écarts" mismatches[1:min(end, 20)]
        end

        @testset "le comparateur voit 1 ULP et un indice faux" begin
            reference = make_const_state(8, 3, [1.0, -2.0, 3.0], [2, 5, 7])
            exp = record_tokens(reference)

            mutated = make_const_state(8, 3, [1.0, -2.0, 3.0], [2, 5, 7])
            mutated.array[5] = reinterpret(Float64,
                reinterpret(UInt64, mutated.array[5]) + 1)
            @test !isempty(compare_records(exp, record_tokens(mutated)))

            shifted = make_const_state(8, 3, [1.0, -2.0, 3.0], [2, 6, 7])
            @test !isempty(compare_records(exp, record_tokens(shifted)))
        end

        @testset "couverture des gardes (branches atteintes)" begin
            # `clear!` creux : seule la liste est zérotée ; la valeur dense qui
            # traîne prouve le chemin creux.
            v = make_const_state(10, 3, [1.0, 2.0, 3.0], [1, 2, 3])
            v.array[9] = 4.0
            clear!(v)
            @test v.array[9] == 4.0 && all(iszero, v.array[1:3])
            # `clear!` dense : remplissage complet.
            v = make_const_state(10, 4, [1.0, 2.0, 3.0, 4.0], [1, 2, 3, 4])
            v.array[9] = 4.0
            clear!(v)
            @test all(iszero, v.array)
            # `reIndex!` écarté (count = 1 ≤ 0.1×10) : la valeur hors liste reste.
            v = make_const_state(10, 1, [1.0], [2])
            v.array[9] = 5.0
            reIndex!(v)
            @test v.count == 1 && v.index[1] == 2
            # `reIndex!` reconstruit (count = 2 > 0.1×10).
            v.count = 2
            reIndex!(v)
            @test v.count == 2 && v.index[1:2] == [2, 9]
        end
    end
end
