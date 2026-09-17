using Test
using TinyHiGHS

@testset "TinyHiGHS — LP File I/O (read_lp / write_lp / solve_lp)" begin
    # Problème test :
    # Minimize obj: 2 x1 + 3 x2 - x3
    # Subject To
    #  c1: x1 + x2 <= 10
    #  c2: 2 x1 - x3 = 5
    #  c3: x2 + x3 >= 1
    # Bounds
    #  0 <= x1 <= 8
    #  x2 >= 0
    #  -2 <= x3 <= 15
    # End

    lp_str = """
\\ Test Problem
Minimize
 obj: 2 x1 + 3 x2 - 1 x3
Subject To
 c1: 1 x1 + 1 x2 <= 10
 c2: 2 x1 - 1 x3 = 5
 c3: 1 x2 + 1 x3 >= 1
Bounds
 0 <= x1 <= 8
 x2 >= 0
 -2 <= x3 <= 15
End
"""
    lp = read_lp(IOBuffer(lp_str))
    @test lp.num_col == 3
    @test lp.num_row == 3
    @test lp.sense == TinyHiGHS.kMinimize

    # Résolution via solve!
    engine = SimplexEngine(lp)
    status = solve!(engine)
    @test status == TinyHiGHS.kOptimal
    @test engine.model_status == TinyHiGHS.kOptimal
    obj1 = engine.info.primal_objective_value

    # Résolution via l'API one-liner solve_lp
    status_direct, obj_direct, _ = solve_lp(lp)
    @test status_direct == TinyHiGHS.kOptimal
    @test obj_direct ≈ obj1 atol=1e-12

    # Test round-trip write_lp -> read_lp
    buf = IOBuffer()
    write_lp(buf, lp)
    str_out = String(take!(buf))
    lp_roundtrip = read_lp(IOBuffer(str_out))

    @test lp_roundtrip.num_col == lp.num_col
    @test lp_roundtrip.num_row == lp.num_row

    status_rt, obj_rt, _ = solve_lp(lp_roundtrip)
    @test status_rt == TinyHiGHS.kOptimal
    @test obj_rt ≈ obj1 atol=1e-12
end
