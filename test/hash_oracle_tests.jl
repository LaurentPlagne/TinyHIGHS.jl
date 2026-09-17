# Oracle C++ — `HighsHashHelpers` : le hachage de base doit être reproduit
# bit-à-bit (il décide de la détection de cyclage du simplexe).
using Printf

const HASH_ORACLE_BIN = abspath(joinpath(@__DIR__, "..", "oracle", "build",
    "hash_oracle"))

"""Séquence de référence du port, dans l'ordre exact de `hash_oracle.cpp`."""
function hash_sequence()
    hashes = String[]
    hash = UInt64(0)
    for index ∈ (0, 1, 63, 64, 65, 127, 128, 200, 1000)
        hash = TinyHiGHS.sparse_combine(hash, index)
        push!(hashes, @sprintf("%016x", hash))
    end
    for index ∈ (1000, 0, 128, 63, 65)
        hash = TinyHiGHS.sparse_inverse_combine(hash, index)
        push!(hashes, @sprintf("%016x", hash))
    end
    for k ∈ 0:19
        hash = TinyHiGHS.sparse_inverse_combine(hash, (k * 37) % 300)
        hash = TinyHiGHS.sparse_combine(hash, (k * 53 + 7) % 300)
        push!(hashes, @sprintf("%016x", hash))
    end
    return hashes
end

if !isfile(HASH_ORACLE_BIN)
    @info "oracle HighsHashHelpers absent — tests ignorés (nécessite oracle/build.sh)"
else
@testset "oracle C++ — HighsHashHelpers (source gelée)" begin
    expected = hash_sequence()
    got = split(read(`$HASH_ORACLE_BIN`, String))
    @test length(expected) == length(got)
    mismatches = ["position $i : $(expected[i]) ≠ oracle $(got[i])"
                  for i ∈ eachindex(expected) if expected[i] != got[i]]
    @test isempty(mismatches)
    isempty(mismatches) || @info "écarts hachage" mismatches[1:min(end, 8)]
    # Sensibilité : une combinaison et son inverse ne redonnent pas le même
    # hachage (l'inverse doit revenir à l'état antérieur, pas rester).
    hash0 = UInt64(0)
    hash1 = TinyHiGHS.sparse_combine(hash0, 7)
    hash2 = TinyHiGHS.sparse_inverse_combine(hash1, 7)
    @test hash2 == hash0
    @test hash1 != hash0
end
end
