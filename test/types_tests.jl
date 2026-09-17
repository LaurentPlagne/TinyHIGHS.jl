@testset "types — SimplexBasis / SimplexInfo" begin
    basis = SimplexBasis(4, 3)
    @test length(basis.basicIndex) == 3
    @test length(basis.nonbasicFlag) == 7 && length(basis.nonbasicMove) == 7
    @test all(iszero, basis.basicIndex) && all(iszero, basis.nonbasicFlag)

    # `nonbasicMove`/`nonbasicFlag` sont des valeurs encodées : -1 et 1 se
    # conservent tels quels (aucun décalage d'indice).
    basis.nonbasicMove[1] = Int8(-1)
    basis.nonbasicMove[2] = Int8(1)
    basis.nonbasicFlag[3] = Int8(1)
    @test basis.nonbasicMove[1:2] == Int8[-1, 1]
    @test basis.nonbasicFlag[3] == Int8(1)

    setup!(basis, 2, 5)
    @test length(basis.basicIndex) == 5 && length(basis.nonbasicFlag) == 7
    clear!(basis)
    @test isempty(basis.basicIndex) && isempty(basis.nonbasicFlag) &&
          isempty(basis.nonbasicMove)

    info = SimplexInfo(4, 3)
    @test length(info.workCost) == 7 && length(info.workDual) == 7
    @test length(info.baseLower) == 3 && length(info.baseValue) == 3
    @test all(iszero, info.workDual) && info.num_primal_infeasibilities == -1
    # Champs M3a : shifts vides, compteurs et densités d'`initialiseControl`.
    @test length(info.workLowerShift) == 7 && length(info.workUpperShift) == 7
    @test info.num_basic_logicals == 0 && !info.costs_perturbed
    @test info.primal_col_density == 0.0 && info.dual_col_density == 1.0
    info.workCost[1] = 2.0
    info.num_dual_infeasibilities = 5
    setup!(info, 2, 2)
    @test length(info.workCost) == 4 && info.workCost[1] == 0.0
    @test length(info.workLowerShift) == 4 && info.dual_col_density == 1.0
    @test info.num_dual_infeasibilities == -1
    clear!(info)
    @test isempty(info.workCost) && isempty(info.baseValue)
    @test isempty(info.workLowerShift) && isempty(info.workUpperShift)

    @test (@inferred SimplexBasis(1, 1)) isa SimplexBasis
    @test (@inferred SimplexInfo(1, 1)) isa SimplexInfo
end
