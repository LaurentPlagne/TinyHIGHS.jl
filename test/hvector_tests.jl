"""Vecteur dense aléatoire décrit comme un `HVector` : indices non triés."""
function random_hvector(rng::AbstractRNG, n::Int; density::Float64=0.4)
    v = HVector(n)
    positions = shuffle(rng, 1:n)[1:round(Int, density * n)]
    for (k, i) ∈ enumerate(positions)
        v.index[k] = i
        v.array[i] = randn(rng)
    end
    v.count = length(positions)
    return v
end

"""Valeurs attendues d'un `HVector`, indexées par position (0 = non listée)."""
function dense_values(v::HVector)
    values = zeros(v.size)
    for i ∈ 1:v.count
        values[v.index[i]] = v.array[v.index[i]]
    end
    return values
end

@testset "TinyHiGHS M0 — HVector" begin
    @testset "setup! / clear!" begin
        v = HVector(10)
        @test (v.size, v.count, v.synthetic_tick, v.packFlag, v.packCount) ==
              (10, 0, 0.0, false, 0)
        @test length(v.array) == 10 && all(iszero, v.array)
        @test length(v.cwork) == 10 + 6400 && length(v.iwork) == 40
        setup!(v, 7)
        @test v.size == 7 && length(v.array) == 7 && length(v.cwork) == 7 + 6400

        v.array[2] = 3.0
        v.array[5] = -1.0
        v.index[1] = 2
        v.index[2] = 5
        v.count = 2
        v.synthetic_tick = 4.0
        v.packFlag = true
        clear!(v)
        @test all(iszero, v.array)
        @test (v.count, v.synthetic_tick, v.packFlag) == (0, 0.0, false)

        # Plus de 30 % de la taille : remplissage complet, même si la liste est
        # incomplète (le vecteur est traité comme dense).
        v.array[6] = 1.0
        v.count = 4
        clear!(v)
        @test all(iszero, v.array)
    end

    @testset "tight!" begin
        v = HVector(6)
        v.array[1] = 1e-15
        v.array[2] = -1e-15
        v.array[3] = 1e-13
        v.array[4] = -3.0
        v.index[1:4] .= (1, 2, 3, 4)
        v.count = 4
        tight!(v)
        @test v.array[1] == 0.0 && v.array[2] == 0.0
        @test v.array[3] == 1e-13 && v.array[4] == -3.0
        @test v.index[1:2] == [3, 4] && v.count == 2

        # count < 0 : indices inconnus, balayage complet.
        w = HVector(6)
        w.array[1] = 1e-15
        w.array[3] = 2.0
        w.count = -1
        tight!(w)
        @test w.array[1] == 0.0 && w.array[3] == 2.0
    end

    @testset "reIndex!" begin
        v = HVector(10)
        v.array[2] = 1.0
        v.array[7] = -1.0
        v.count = -1
        reIndex!(v)
        @test v.count == 2 && v.index[1:2] == [2, 7]

        # Liste valide et creuse (count ≤ 10 % de la taille) : la garde écarte
        # le balayage, même si une valeur traîne hors liste (contrat de la
        # source : la liste couvre les nonzeros).
        v.count = 1
        v.array[9] = 5.0
        reIndex!(v)
        @test v.count == 1 && v.index[1] == 2

        # Au-delà de 10 % de la taille : reconstruction.
        v.count = 5
        reIndex!(v)
        @test v.count == 3 && v.index[1:3] == [2, 7, 9]
    end

    @testset "pack!" begin
        v = HVector(5)
        v.index[1:2] .= (4, 2)
        v.array[4] = 1.5
        v.array[2] = -2.5
        v.count = 2
        pack!(v)                      # packFlag faux : rien
        @test v.packCount == 0
        v.packFlag = true
        pack!(v)
        @test v.packFlag == false && v.packCount == 2
        @test v.packIndex[1:2] == [4, 2] && v.packValue[1:2] == [1.5, -2.5]
    end

    @testset "copy!" begin
        from = HVector(4)
        from.index[1:2] .= (3, 1)
        from.array[3] = 7.0
        from.array[1] = -1.0
        from.count = 2
        from.synthetic_tick = 9.0
        # La liste de la destination doit couvrir ses nonzeros (même contrat
        # que la source) : `clear!` s'appuie dessus.
        to = HVector(4)
        to.index[1] = 2
        to.array[2] = 4.0
        to.count = 1
        copy!(to, from)
        @test to.count == 2 && to.index[1:2] == [3, 1]
        @test to.array == from.array
        @test to.synthetic_tick == 9.0
    end

    @testset "norm2" begin
        rng = MersenneTwister(20260916)
        v = random_hvector(rng, 40)
        expected = sum(v.array[v.index[i]]^2 for i ∈ 1:v.count; init=0.0)
        @test norm2(v) == expected
    end

    @testset "saxpy!" begin
        v = HVector(5)
        v.index[1] = 2
        v.array[2] = 1.0
        v.count = 1
        pivot = HVector(5)
        pivot.index[1:3] .= (2, 4, 5)
        pivot.array[2] = 2.0
        pivot.array[4] = 3.0
        pivot.array[5] = 1e-15
        pivot.count = 3
        saxpy!(v, 2.0, pivot)
        @test v.array[2] == 5.0                    # déjà listé, pas re-ajouté
        @test v.array[4] == 6.0 && v.index[2] == 4
        @test v.array[5] == kHighsZero             # sous kHighsTiny
        @test v.count == 3
        @test_throws ArgumentError saxpy!(v, 1.0, v)
    end

    @testset "valeurs aléatoires : tight! + reIndex! contre référence naïve" begin
        rng = MersenneTwister(7)
        for _ ∈ 1:20
            n = rand(rng, 5:60)
            v = random_hvector(rng, n; density=0.5)
            v.array[v.index[1:max(1, v.count ÷ 4)]] .*= 1e-16
            expected = dense_values(v)
            expected[abs.(expected) .< kHighsTiny] .= 0.0
            tight!(v)
            @test dense_values(v) == expected
            @test all(abs(v.array[v.index[i]]) >= kHighsTiny for i ∈ 1:v.count)
        end
    end

    @testset "allocations nulles sur chemins actifs (après chauffe)" begin
        # Chauffe (compilation) sur des états jetables.
        warm = random_hvector(MersenneTwister(0), 32; density=0.5)
        warmp = random_hvector(MersenneTwister(1), 32; density=0.5)
        tight!(warm)
        reIndex!(warm)
        clear!(warm)
        pack!(warm)
        saxpy!(warm, 1.0, warmp)
        copy!(HVector(32), warm)

        # `tight!` actif : des valeurs sous kHighsTiny.
        v = random_hvector(MersenneTwister(2), 64; density=0.2)
        for i ∈ 1:min(3, v.count)
            v.array[v.index[i]] = 1e-16
        end
        @test (@allocated tight!(v)) == 0

        # `reIndex!` actif : reconstruction par balayage complet.
        v = random_hvector(MersenneTwister(3), 64; density=0.2)
        v.count = -1
        @test (@allocated reIndex!(v)) == 0

        # `pack!` actif : packFlag armé, chemin de copie réellement exécuté.
        v = random_hvector(MersenneTwister(4), 64; density=0.2)
        v.packFlag = true
        @test (@allocated pack!(v)) == 0

        # `clear!` dense (remplissage complet) puis creux (zéro selon la liste).
        v = random_hvector(MersenneTwister(5), 64; density=0.5)
        @test v.count > 0.3 * v.size
        @test (@allocated clear!(v)) == 0
        v = random_hvector(MersenneTwister(6), 64; density=0.2)
        @test v.count <= 0.3 * v.size
        @test (@allocated clear!(v)) == 0

        # `copy!` actif.
        to = random_hvector(MersenneTwister(7), 64; density=0.2)
        from = random_hvector(MersenneTwister(8), 64; density=0.2)
        @test (@allocated copy!(to, from)) == 0

        # `saxpy!` actif.
        v = random_hvector(MersenneTwister(9), 64; density=0.2)
        pivot = random_hvector(MersenneTwister(10), 64; density=0.5)
        @test (@allocated saxpy!(v, 1.0, pivot)) == 0

        # `norm2` actif.
        @test (@allocated norm2(v)) == 0
    end

    @testset "inférence" begin
        v = @inferred HVector(8)
        pivot = @inferred HVector(8)
        @test (@inferred clear!(v)) === v
        @test (@inferred tight!(v)) === v
        @test (@inferred reIndex!(v)) === v
        @test (@inferred pack!(v)) === v
        @test (@inferred saxpy!(v, 1.0, pivot)) === v
        @test (@inferred norm2(v)) isa Float64
    end
end
