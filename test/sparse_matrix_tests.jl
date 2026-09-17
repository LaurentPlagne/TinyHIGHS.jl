# Tests M2a — `SparseMatrix` : vues cohérentes, produits, pricing.
# Oracle C++ dédié encore à écrire (cf. plan §4.1) ; ici, références naïves
# en double précision et contrôle de l'ordre des sommes quand il est connu.
using Random

function random_sparse(rng::AbstractRNG, num_row::Int, num_col::Int)
    a_start = Int[1]
    a_index = Int[]
    a_value = Float64[]
    for j ∈ 1:num_col
        rows = sort(shuffle(rng, 1:num_row)[1:rand(rng, 1:3)])
        for r ∈ rows
            push!(a_index, r)
            push!(a_value, randn(rng))
        end
        push!(a_start, length(a_index) + 1)
    end
    return SparseMatrix(num_col, num_row, a_start, a_index, a_value)
end

function dense_matrix(m::SparseMatrix)
    A = zeros(m.num_row, m.num_col)
    if is_colwise(m)
        for j ∈ 1:m.num_col
            for k ∈ m.start[j]:(m.start[j + 1] - 1)
                A[m.index[k], j] += m.value[k]
            end
        end
    else
        for i ∈ 1:m.num_row
            for k ∈ m.start[i]:(m.start[i + 1] - 1)
                A[i, m.index[k]] += m.value[k]
            end
        end
    end
    return A
end

@testset "M2a — SparseMatrix (vues, produits, pricing)" begin
    rng = MersenneTwister(20260917)
    m = random_sparse(rng, 8, 6)
    A = dense_matrix(m)
    @test num_nz(m) == length(m.index)
    @test is_colwise(m) && !is_rowwise(m)

    @testset "vues rowwise/colwise" begin
        mr = copy(m)
        ensure_rowwise!(mr)
        @test is_rowwise(mr) && num_nz(mr) == num_nz(m)
        @test dense_matrix(mr) == A
        for i ∈ 1:8
            idx, val = get_row(mr, i)
            @test length(idx) == length(val)
            @test all(A[i, j] == 0 || any(idx .== j) for j ∈ 1:6)
        end
        ensure_colwise!(mr)
        @test is_colwise(mr) && dense_matrix(mr) == A
    end

    @testset "produits" begin
        x = randn(rng, 6)
        y = randn(rng, 8)
        result = zeros(8)
        product!(result, m, x)
        @test result ≈ A * x
        mr = copy(m)
        ensure_rowwise!(mr)
        result2 = zeros(8)
        product!(result2, mr, x)
        @test result2 ≈ A * x
        rt = zeros(6)
        product_transpose!(rt, m, y)
        @test rt ≈ A' * y
        rt2 = zeros(6)
        product_transpose!(rt2, mr, y)
        @test rt2 ≈ A' * y
        y2 = randn(rng, 8)
        expected = copy(y2) + 2.5 * (A * x)
        alpha_product_plus_y!(y2, 2.5, m, x)
        @test y2 ≈ expected
        y3 = randn(rng, 6)
        expected3 = copy(y3) + 2.5 * (A' * y)
        alpha_product_plus_y!(y3, 2.5, m, y, transpose=true)
        @test y3 ≈ expected3
    end

    @testset "computeDot / collectAj" begin
        z = randn(rng, 8)
        for j ∈ 1:6
            @test compute_dot(m, z, j) == sum(z[m.index[k]] * m.value[k]
                                              for k ∈ m.start[j]:(m.start[j + 1] - 1);
                init=0.0)
        end
        # Colonne logique : index ≥ num_col+1 → valeur directe.
        @test compute_dot(m, z, 6 + 3) == z[3]

        v = HVector(8)
        v.count = 0
        collect_aj!(m, v, 2, 1.5)
        # Référence naïve dans le même ordre : seules les positions de la
        # colonne sont touchées (une position intacte reste 0.0).
        expected = zeros(8)
        indices = Int[]
        for k ∈ m.start[2]:(m.start[3] - 1)
            iRow = m.index[k]
            value1 = expected[iRow] + 1.5 * m.value[k]
            value0 = expected[iRow]
            expected[iRow] = abs(value1) < kHighsTiny ? kHighsZero : value1
            value0 == 0.0 && push!(indices, iRow)
        end
        @test v.array == expected
        @test sort(v.index[1:v.count]) == sort(indices)
        v_logique = HVector(8)
        v_logique.count = 0
        collect_aj!(m, v_logique, 6 + 2, 0.5)
        @test v_logique.array[2] == 0.5
    end

    @testset "priceByColumn" begin
        column = HVector(8)
        column.count = -1
        column.array .= randn(rng, 8)
        result = HVector(6)
        result.count = 0
        price_by_column!(m, result, column)
        expected = [sum(column.array[m.index[k]] * m.value[k]
                        for k ∈ m.start[j]:(m.start[j + 1] - 1); init=0.0)
                    for j ∈ 1:6]
        for j ∈ 1:6
            @test result.array[j] == expected[j]
        end
        @test result.index[1:result.count] ==
              [j for j ∈ 1:6 if abs(expected[j]) > kHighsTiny]
    end

    @testset "priceByRow / priceByRowWithSwitch" begin
        mr = copy(m)
        ensure_rowwise!(mr)
        col = HVector(8)
        rows = sort(shuffle(rng, 1:8)[1:3])
        for (k, i) ∈ enumerate(rows)
            col.index[k] = i
            col.array[i] = randn(rng)
        end
        col.count = length(rows)
        ref = zeros(6)
        for ix ∈ 1:col.count
            iRow = col.index[ix]
            mult = col.array[iRow]
            for k ∈ mr.start[iRow]:(mr.start[iRow + 1] - 1)
                iCol = mr.index[k]
                v0 = ref[iCol]
                v1 = v0 + mult * mr.value[k]
                ref[iCol] = abs(v1) < kHighsTiny ? kHighsZero : v1
            end
        end
        result = HVector(6)
        result.count = 0
        price_by_row!(mr, result, col)
        @test result.array == ref
        # L'ordre de `result.index` est l'ordre de première touche, pas le tri.
        @test sort(result.index[1:result.count]) ==
              [j for j ∈ 1:6 if !iszero(ref[j])]

        # Bascule immédiate (switch_density = 0) : même résultat à l'arrondi.
        result2 = HVector(6)
        result2.count = 0
        price_by_row_with_switch!(mr, result2, col, -Inf, 1, 0.0)
        @test result2.array ≈ result.array
        @test sort(result2.index[1:result2.count]) ==
              sort(result.index[1:result.count])
    end
end
