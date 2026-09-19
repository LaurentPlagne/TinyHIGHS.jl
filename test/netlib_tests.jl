const NETLIB_TEST_REFERENCE = Dict{String,Float64}(
    "adlittle.lp" => 2.2549496316e5,
    "afiro.lp" => -4.6475314286e2,
    "blend.lp" => -3.0812149846e1,
    "sc50a.lp" => -6.4575077059e1,
    "sc50b.lp" => -7.0e1,
    "share2b.lp" => -4.1573224074e2,
)

@testset "Netlib — curated smoke corpus" begin
    root = normpath(joinpath(@__DIR__, "..", "instances", "netlib"))
    @test isdir(root)
    for (filename, reference) in NETLIB_TEST_REFERENCE
        path = joinpath(root, filename)
        @test isfile(path)
        isfile(path) || continue
        status, objective, engine = solve_lp(path)
        @test status == TinyHiGHS.kOptimal
        @test objective ≈ reference rtol=2e-7 atol=2e-7 * max(1.0, abs(reference))
        @test engine.lp.num_col >= 0
        @test engine.lp.num_row >= 0
    end
end
