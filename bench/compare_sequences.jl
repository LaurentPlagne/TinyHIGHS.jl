# ==============================================================================
# Benchmark comparatif de SÉQUENCES (Warm-Start) : TinyHiGHS vs HiGHS C++
# ==============================================================================

using Printf
using TinyHiGHS

const SEQ_ROOT = abspath(joinpath(@__DIR__, "..", "instances", "sequences"))
const SEQ_ORACLE_BIN = abspath(joinpath(@__DIR__, "..", "oracle", "build", "sequence_oracle"))

"""
Rejoue une séquence avec TinyHiGHS en pur warm-start (moteur unique réutilisé).
"""
function replay_sequence_tinyhighs(base_lp_path::String, ops_path::String)
    lp = read_lp(base_lp_path)
    engine = SimplexEngine(lp)
    
    solves_count = 0
    total_iters = 0
    last_obj = 0.0
    
    t0 = time_ns()
    for line in eachline(ops_path)
        toks = split(line)
        isempty(toks) && continue
        op = toks[1]
        
        if op == "change_col_bounds"
            c = parse(Int, toks[2])
            lo = parse(Float64, toks[3])
            up = parse(Float64, toks[4])
            change_col_bounds!(engine, c, lo, up)
        elseif op == "change_cols_bounds"
            n = parse(Int, toks[2])
            pos = 3
            for _ in 1:n
                c = parse(Int, toks[pos])
                lo = parse(Float64, toks[pos+1])
                up = parse(Float64, toks[pos+2])
                change_col_bounds!(engine, c, lo, up)
                pos += 3
            end
        elseif op == "change_row_bounds"
            r = parse(Int, toks[2])
            lo = parse(Float64, toks[3])
            up = parse(Float64, toks[4])
            change_row_bounds!(engine, r, lo, up)
        elseif op == "change_rows_bounds"
            n = parse(Int, toks[2])
            pos = 3
            for _ in 1:n
                r = parse(Int, toks[pos])
                lo = parse(Float64, toks[pos+1])
                up = parse(Float64, toks[pos+2])
                change_row_bounds!(engine, r, lo, up)
                pos += 3
            end
        elseif op == "change_cols_cost"
            n = parse(Int, toks[2])
            pos = 3
            cols = Int[]
            costs = Float64[]
            for _ in 1:n
                push!(cols, parse(Int, toks[pos]))
                push!(costs, parse(Float64, toks[pos+1]))
                pos += 2
            end
            change_cols_cost!(engine, cols, costs)
        elseif op == "solve"
            solves_count += 1
            initialise_for_solve!(engine)
            if engine.model_status != TinyHiGHS.kOptimal
                solve!(DualSolver(engine))
            else
                restore_scale!(engine)
            end
            total_iters += engine.iteration_count
            last_obj = engine.info.primal_objective_value
        elseif op == "end"
            break
        end
    end
    elapsed_ms = (time_ns() - t0) / 1e6
    return (time_ms=elapsed_ms, solves=solves_count, iters=total_iters, last_obj=last_obj)
end

"""
Rejoue une séquence avec HiGHS C++ (via sequence_oracle) en pur warm-start.
"""
function replay_sequence_highs_c(base_lp_path::String, ops_path::String)
    isfile(SEQ_ORACLE_BIN) || error("sequence_oracle binaire non trouvé à $SEQ_ORACLE_BIN")
    lp = read_lp(base_lp_path)
    m = lp.a_matrix
    
    # Préparation du flux d'entrée pour sequence_oracle (format M3/M5)
    input = IOBuffer()
    println(input, lp.num_col, " ", lp.num_row, " ", length(m.index))
    println(input, join(lp.col_cost, " "))
    println(input, join(lp.col_lower, " "))
    println(input, join(lp.col_upper, " "))
    println(input, join(lp.row_lower, " "))
    println(input, join(lp.row_upper, " "))
    println(input, join(m.start .- 1, " "))
    println(input, join(m.index .- 1, " "))
    println(input, join(m.value, " "))
    println(input, Int(lp.sense), " 0.0")
    
    for l in eachline(ops_path)
        println(input, l)
    end
    
    in_str = String(take!(input))
    out = IOBuffer()
    
    t0 = time_ns()
    proc = run(pipeline(`$SEQ_ORACLE_BIN 0 -1 0.0 0 -1.0 -1 -1.0 -1 0`; stdin=IOBuffer(in_str), stdout=out))
    elapsed_ms = (time_ns() - t0) / 1e6
    
    total_iters = 0
    solves_count = 0
    last_obj = 0.0
    for l in eachline(IOBuffer(String(take!(out))))
        toks = split(l)
        isempty(toks) && continue
        if toks[1] == "solve"
            solves_count += 1
            last_obj = parse(Float64, toks[4])
            total_iters += parse(Int, toks[5])
        end
    end
    return (time_ms=elapsed_ms, solves=solves_count, iters=total_iters, last_obj=last_obj)
end

function run_all_sequence_benchmarks()
    println("="^95)
    println(" BENCHMARK COMPARATIF DES SUITES DE LP (WARM-START) : TinyHiGHS.jl vs HiGHS C++")
    println("="^95)
    println("Oracle C++ HiGHS : ", isfile(SEQ_ORACLE_BIN) ? SEQ_ORACLE_BIN : "Non compilé")
    println()
    
    sequences = ["sequence_small", "sequence_medium"]
    
    @printf("%-18s | %-7s | %-14s | %-14s | %-10s | %-12s\n",
            "Suite de LPs", "Solves", "TinyHiGHS (tot)", "HiGHS C++ (tot)", "Speedup", "Temps/solve (Tiny)")
    println("-"^95)
    
    for seq_name in sequences
        dir = joinpath(SEQ_ROOT, seq_name)
        base_lp = joinpath(dir, "base.lp")
        ops_file = joinpath(dir, "operations.txt")
        
        isfile(base_lp) && isfile(ops_file) || continue
        
        # Warmup
        replay_sequence_tinyhighs(base_lp, ops_file)
        
        # Benchmark TinyHiGHS
        res_tiny = replay_sequence_tinyhighs(base_lp, ops_file)
        
        # Benchmark HiGHS C++
        if isfile(SEQ_ORACLE_BIN)
            res_highs = replay_sequence_highs_c(base_lp, ops_file)
            ratio = res_highs.time_ms / max(res_tiny.time_ms, 0.001)
            us_per_solve = (res_tiny.time_ms * 1000.0) / res_tiny.solves
            @printf("%-18s | %7d | %11.2f ms | %11.2f ms | %9.2fx | %9.1f µs/solve\n",
                    seq_name, res_tiny.solves, res_tiny.time_ms, res_highs.time_ms, ratio, us_per_solve)
        else
            us_per_solve = (res_tiny.time_ms * 1000.0) / res_tiny.solves
            @printf("%-18s | %7d | %11.2f ms | %14s | %10s | %9.1f µs/solve\n",
                    seq_name, res_tiny.solves, res_tiny.time_ms, "N/A", "-", us_per_solve)
        end
    end
    println("="^95)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_sequence_benchmarks()
end
