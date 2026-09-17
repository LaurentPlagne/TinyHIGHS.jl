# Oracle C++ — `HighsRandom` : la séquence de tirages du port doit reproduire
# celle de la source (elle décide de l'ordre de balayage de CHUZR et des shifts).
using Printf

const RANDOM_ORACLE_BIN = abspath(joinpath(@__DIR__, "..", "oracle", "build",
    "random_oracle"))

"""Séquence de référence du port, dans l'ordre exact de `random_oracle.cpp`."""
function random_sequence(rng::TinyHiGHS.HighsRandom)
    tokens = String[]
    for sup ∈ (1000, 1, 0, 2, 2147483647, 65536, 3)
        for _ ∈ 1:4
            push!(tokens, string(TinyHiGHS.integer(rng, sup)))
        end
    end
    for _ ∈ 1:6
        push!(tokens, @sprintf("%a", TinyHiGHS.fraction(rng)))
    end
    # Les deux `shuffle` permutent les mêmes 8 valeurs 0..7 (seul l'indexage
    # des emplacements diffère) : comparer les valeurs telles quelles.
    data = collect(0:7)
    TinyHiGHS.shuffle!(rng, data)
    append!(tokens, string.(data))
    TinyHiGHS.initialise!(rng, 42)
    for _ ∈ 1:6
        push!(tokens, string(TinyHiGHS.integer(rng, 100)))
    end
    for _ ∈ 1:6
        push!(tokens, @sprintf("%a", TinyHiGHS.fraction(rng)))
    end
    return tokens
end

if !isfile(RANDOM_ORACLE_BIN)
    @info "oracle HighsRandom absent — tests ignorés (nécessite oracle/build.sh)"
else
@testset "oracle C++ — HighsRandom (source gelée)" begin
    expected = random_sequence(TinyHiGHS.HighsRandom(0))
    got = split(read(`$RANDOM_ORACLE_BIN`, String))
    @test length(expected) == length(got)
    mismatches = ["position $i : $(expected[i]) ≠ oracle $(got[i])"
                  for i ∈ eachindex(expected) if expected[i] != got[i]]
    @test isempty(mismatches)
    isempty(mismatches) || @info "écarts RNG" mismatches[1:min(end, 8)]
    # Sensibilité : un décalage d'un cran dans la suite est vu.
    shifted = vcat(expected[2:end], expected[end])
    @test any(expected[i] != shifted[i] for i ∈ eachindex(expected))
end
end
