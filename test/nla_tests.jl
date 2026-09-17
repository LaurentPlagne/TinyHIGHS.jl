# Tests M2b — `Nla` : solves échelonnés et conventions d'échelles.
# Oracle C++ dédié à écrire (nécessite un contexte type HEkk) ; ici, contrôles
# par l'algèbre : `ftran`/`btran` doivent résoudre le système NON échelonné.
using Random

@testset "M2b — Nla (solves échelonnés, conventions)" begin
    rng = MersenneTwister(20260919)
    n = 6
    a_start = Int[1]
    a_index = Int[]
    a_value = Float64[]
    B = zeros(n, n)
    for j ∈ 1:n
        entries = sort(shuffle(rng, 1:n)[1:3])
        for r ∈ entries
            v = r == j ? 4.0 + rand(rng) : randn(rng)
            push!(a_index, r)
            push!(a_value, v)
            B[r, j] += v
        end
        push!(a_start, length(a_index) + 1)
    end
    basic = collect(1:n)
    col_scale = exp.(randn(rng, n) * 0.5)
    row_scale = exp.(randn(rng, n) * 0.5)
    scale = Scale(col_scale, row_scale)
    # Le facteur est construit sur la matrice ÉCHELONNÉE (contrat de la NLA).
    m_scaled = SparseMatrix(n, n, a_start, a_index, a_value)
    apply_scale!(m_scaled, scale)
    factor = HFactor(n, n, n, m_scaled.start, m_scaled.index, m_scaled.value,
        basic)
    nla = Nla(factor, n, n; scale=scale)
    @test invert!(nla) == 0

    # `build!` permute `basic_index` ; les solutions sont dans cet ordre.
    bp = nla.basic_index
    B_perm = B[:, bp]
    @testset "facteurs d'échelle" begin
        @test variable_scale_factor(nla, 3) == col_scale[3]
        @test variable_scale_factor(nla, n + 2) == 1.0 / row_scale[2]
        @test basic_col_scale_factor(nla, 4) == col_scale[bp[4]]
        aq = HVector(n)
        aq.count = 1
        aq.index[1] = 4
        aq.array[4] = 1.25                       # le pivot est lu à row_out
        expected = 1.25 * col_scale[2] / col_scale[bp[4]]
        @test pivot_in_scaled_space(nla, aq, 2, 4) == expected
    end

    @testset "ftran / btran non échelonnés" begin
        rhs = randn(rng, n)
        v = HVector(n)
        v.count = -1
        v.array .= rhs
        ftran!(nla, v, 1.0)
        @test maximum(abs.(B_perm * v.array .- rhs)) < 1e-10
        w = HVector(n)
        w.count = -1
        w.array .= rhs
        btran!(nla, w, 1.0)
        @test maximum(abs.(B_perm' * w.array .- rhs)) < 1e-10
    end

    @testset "sans échelle : identique au facteur" begin
        factor2 = HFactor(n, n, n, a_start, a_index, a_value, basic)
        nla2 = Nla(factor2, n, n)
        build!(factor2)
        rhs = randn(rng, n)
        v1 = HVector(n)
        v1.count = -1
        v1.array .= rhs
        ftran!(nla2, v1, 1.0)
        v2 = HVector(n)
        v2.count = -1
        v2.array .= rhs
        ftranCall!(factor2, v2, 1.0)
        @test v1.array == v2.array
    end

    @testset "rowEp2NormInScaledSpace" begin
        row_ep = HVector(n)
        row_ep.count = -1
        row_ep.array .= randn(rng, n)
        iRow = 3
        expected = sum((row_ep.array[r] / (row_scale[r] * col_scale[bp[iRow]]))^2
                       for r ∈ 1:n; init=0.0)
        @test row_ep_2norm_in_scaled_space(nla, iRow, row_ep) ≈ expected
    end

    @testset "transformForUpdate" begin
        aq = HVector(n)
        aq.count = 2
        aq.index[1:2] .= (2, 4)
        aq.array[2] = 1.5
        aq.array[4] = -0.5
        aq.packCount = 2
        aq.packIndex[1:2] .= (2, 4)
        aq.packValue[1:2] .= (1.5, -0.5)
        ep = HVector(n)
        ep.count = 1
        ep.index[1] = 5
        ep.array[5] = 0.75
        ep.packCount = 1
        ep.packIndex[1] = 5
        ep.packValue[1] = 0.75
        variable_in = 2
        row_out = 4
        cq = col_scale[2]
        cp = col_scale[bp[4]]
        transform_for_update!(nla, aq, ep, variable_in, row_out)
        @test aq.packValue[1:2] == [1.5 * cq, -0.5 * cq]
        @test aq.array[4] == -0.5 * cq / cp
        @test ep.packValue[1] == 0.75 / cp
    end
end
