function _pivot_ulp_neighbor(value::Float64, reference::Float64)
    return value == reference || value == prevfloat(reference) ||
           value == nextfloat(reference)
end

@testset "HFactor pivot strategies" begin
    p = 3.25

    @testset "unit pivots are exact" begin
        for strategy ∈ (kPivotBranching, kPivotBranchless, kPivotFdiv)
            value = Val(strategy)
            @test TinyHiGHS.apply_pivot(value, p, 1.0, 1.0) === p
            @test TinyHiGHS.apply_pivot(value, p, -1.0, -1.0) === -p
        end
    end

    @testset "branchless reciprocal is bounded" begin
        pivot = 3.0
        reciprocal = 1.0 / pivot
        reference = p / pivot
        branchless = TinyHiGHS.apply_pivot(Val(kPivotBranchless), p, pivot,
            reciprocal)
        @test _pivot_ulp_neighbor(branchless, reference)
        @test branchless === p * reciprocal
        @test TinyHiGHS.apply_pivot(Val(kPivotBranching), p, pivot,
            reciprocal) === reference
        @test TinyHiGHS.apply_pivot(Val(kPivotFdiv), p, pivot,
            reciprocal) === reference

        rng = MersenneTwister(20260919)
        for _ ∈ 1:1_000
            pivot = 0.25 + 3.5 * rand(rng)
            multiplier = randn(rng)
            reciprocal = 1.0 / pivot
            @test _pivot_ulp_neighbor(
                TinyHiGHS.apply_pivot(Val(kPivotBranchless), multiplier,
                    pivot, reciprocal), multiplier / pivot)
        end
    end

    @testset "reciprocal cache follows updates" begin
        f = HFactor(2, 2, 2, [1, 2, 3], [1, 2], [1.0, 1.0], [1, 2])
        build!(f)
        @test f.u_pivot_inv_value == 1.0 ./ f.u_pivot_value

        aq = HVector(2)
        aq.index[1] = 1
        aq.array[1] = 2.0
        aq.count = 1
        aq.packFlag = true
        pack!(aq)
        ep = HVector(2)
        ep.index[1] = 1
        ep.array[1] = 1.0
        ep.count = 1
        ep.packFlag = true
        pack!(ep)
        update!(f, aq, ep, 1)
        @test f.u_pivot_inv_value == 1.0 ./ f.u_pivot_value
    end

    @testset "same feasibility on a non-unit factor" begin
        matrix = SparseMatrix(2, 2, [1, 3, 5], [1, 2, 1, 2],
            [2.0, 1.0, 1.0, 2.0])
        results = NamedTuple[]
        old_strategy = ACTIVE_PIVOT_STRATEGY[]
        try
            for strategy ∈ (kPivotBranching, kPivotBranchless, kPivotFdiv)
                set_pivot_strategy!(strategy)
                lp = SimplexLp(2, 2, SparseMatrix(matrix.num_col, matrix.num_row,
                    copy(matrix.start), copy(matrix.index), copy(matrix.value)),
                    [1.0, 1.0], [0.0, 0.0], [10.0, 10.0],
                    [1.0, 1.0], [1.0, 1.0])
                engine = SimplexEngine(lp)
                initialise_for_solve!(engine)
                solve!(DualSolver(engine))
                push!(results, (status=engine.model_status,
                    objective=engine.info.primal_objective_value,
                    iterations=engine.iteration_count))
            end
        finally
            set_pivot_strategy!(old_strategy)
        end
        @test all(result -> result.status === TinyHiGHS.kOptimal, results)
        @test all(result -> isapprox(result.objective, 2.0 / 3.0;
            rtol=0.0, atol=8eps(1.0)), results)
        @test length(unique(result.iterations for result ∈ results)) == 1
    end
end
