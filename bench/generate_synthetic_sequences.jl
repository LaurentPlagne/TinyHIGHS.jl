# ==============================================================================
# Synthetic Sequential Warm-Start Benchmark Generator and Runner
# Evaluates Warm-Start vs Cold-Start re-optimization and Zero-Allocation Guarantees
# ==============================================================================

using Printf
using Random
using TinyHiGHS

"""
Generate a synthetic network flow / transportation linear program.
"""
function generate_network_flow_lp(num_sources::Int, num_sinks::Int; seed::Int=42)
    rng = Random.MersenneTwister(seed)
    num_col = num_sources * num_sinks
    num_row = num_sources + num_sinks

    # Costs: random positive transportation costs
    col_cost = rand(rng, num_col) .* 10.0 .+ 1.0

    # Bounds: capacities [0, Inf)
    col_lower = zeros(Float64, num_col)
    col_upper = fill(kHighsInf, num_col)

    # Constraints: supply and demand
    supply = rand(rng, num_sources) .* 50.0 .+ 50.0
    demand = rand(rng, num_sinks) .* 50.0 .+ 50.0
    # Balance total supply and demand
    supply .*= (sum(demand) / sum(supply))

    row_lower = [supply; demand]
    row_upper = copy(row_lower) # Equality constraints

    # Constraint matrix A
    a_start = zeros(Int, num_col + 1)
    a_start[1] = 1
    total_nz = num_col * 2
    a_index = zeros(Int, total_nz)
    a_value = zeros(Float64, total_nz)

    pos = 1
    col = 1
    for s in 1:num_sources
        for t in 1:num_sinks
            # Supply row s: +1
            a_index[pos] = s
            a_value[pos] = 1.0
            pos += 1
            # Demand row num_sources + t: +1
            a_index[pos] = num_sources + t
            a_value[pos] = 1.0
            pos += 1
            col += 1
            a_start[col] = pos
        end
    end

    matrix = SparseMatrix(num_col, num_row, a_start, a_index, a_value)
    return SimplexLp(num_col, num_row, matrix, col_cost, col_lower, col_upper, row_lower, row_upper)
end

"""
Benchmark synthetic warm-start resolve sequence vs cold-start resolves.
"""
function run_synthetic_warmstart_benchmarks(; num_steps::Int=100, seed::Int=123)
    println("="^95)
    println(" SYNTHETIC SEQUENTIAL WARM-START BENCHMARK (TinyHiGHS)")
    println(" Measuring Microsecond Warm-Starts & Zero-Allocation (@allocated) Across $num_steps Resolves")
    println("="^95)

    base_lp = generate_network_flow_lp(10, 15; seed=seed)
    println("Problem dimensions: $(base_lp.num_row) rows x $(base_lp.num_col) columns, $(length(base_lp.a_matrix.index)) nonzeros")
    println()

    rng = Random.MersenneTwister(seed + 1)

    # Pre-generate perturbations for repeatability
    row_pert = [rand(rng, base_lp.num_row) .* 2.0 .- 1.0 for _ in 1:num_steps]
    cost_pert = [rand(rng, base_lp.num_col) .* 0.5 .- 0.25 for _ in 1:num_steps]

    # --- 1. Warm-Start Sequence ---
    engine = SimplexEngine(base_lp)
    initialise_for_solve!(engine)
    solve!(engine) # Initial solve

    warm_times_us = Float64[]
    warm_allocs = Int[]
    total_warm_iters = 0

    for step in 1:num_steps
        # Apply small bounded variations to supply/demand and costs
        for r in 1:base_lp.num_row
            val = max(10.0, engine.lp.row_lower[r] + row_pert[step][r])
            change_row_bounds!(engine, r, val, val)
        end
        for c in 1:base_lp.num_col
            cost = max(0.1, engine.lp.col_cost[c] + cost_pert[step][c])
            engine.lp.col_cost[c] = cost
        end
        update_status!(engine, TinyHiGHS.kLpActionNewCosts)

        # Measure resolve
        t0 = time_ns()
        alloc = @allocated begin
            solve!(engine)
        end
        dt_us = (time_ns() - t0) / 1000.0

        push!(warm_times_us, dt_us)
        push!(warm_allocs, alloc)
        total_warm_iters += engine.iteration_count
    end

    # --- 2. Cold-Start Sequence ---
    cold_times_us = Float64[]
    total_cold_iters = 0

    for step in 1:num_steps
        # Reconstruct updated model from current bounds and costs
        curr_lp = SimplexLp(
            base_lp.num_col, base_lp.num_row, base_lp.a_matrix,
            copy(engine.lp.col_cost), copy(engine.lp.col_lower), copy(engine.lp.col_upper),
            copy(engine.lp.row_lower), copy(engine.lp.row_upper)
        )
        t0 = time_ns()
        c_engine = SimplexEngine(curr_lp)
        solve!(c_engine)
        dt_us = (time_ns() - t0) / 1000.0

        push!(cold_times_us, dt_us)
        total_cold_iters += c_engine.iteration_count
    end

    avg_warm_us = sum(warm_times_us) / num_steps
    min_warm_us = minimum(warm_times_us)
    avg_cold_us = sum(cold_times_us) / num_steps
    speedup = avg_cold_us / avg_warm_us
    zero_alloc_pct = 100.0 * count(==(0), warm_allocs) / num_steps

    @printf("Results over %d sequential re-optimizations:\n", num_steps)
    @printf("  - Cold-Start Average Time : %8.1f µs / resolve (Total: %8.2f ms, %d iterations)\n",
            avg_cold_us, sum(cold_times_us) / 1000.0, total_cold_iters)
    @printf("  - Warm-Start Average Time : %8.1f µs / resolve (Total: %8.2f ms, %d iterations)\n",
            avg_warm_us, sum(warm_times_us) / 1000.0, total_warm_iters)
    @printf("  - Warm-Start Best Time    : %8.1f µs / resolve\n", min_warm_us)
    @printf("  - Warm-Start Speedup      : %8.2fx faster than cold-start\n", speedup)
    @printf("  - Zero-Allocation Rate    : %8.1f%% of warm resolves had @allocated == 0 bytes\n", zero_alloc_pct)
    println("="^95)
    return (speedup=speedup, avg_warm_us=avg_warm_us, avg_cold_us=avg_cold_us, zero_alloc_pct=zero_alloc_pct)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_synthetic_warmstart_benchmarks()
end
