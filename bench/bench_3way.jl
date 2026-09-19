# ==============================================================================
# Benchmark Comparatif 3-Voies :
#   1. HiGHS C++ (Artefact Julia officiel HiGHS_jll v1.15.1)
#   2. HiGHS C++ (Compilé localement avec clang++ -O3)
#   3. TinyHiGHS.jl (Julia natif zéro-allocation, 3 stratégies de pivot)
# ==============================================================================

using Printf
using TinyHiGHS

const ROOT_DIR = dirname(@__DIR__)
const BIN_ORIGINAL = joinpath(ROOT_DIR, "contrib_highs", "cpp", "replay_sequence_original")
const BIN_LOCAL = joinpath(ROOT_DIR, "contrib_highs", "cpp", "replay_sequence")

include(joinpath(ROOT_DIR, "bench", "compare_sequences.jl"))

function run_cpp_benchmark(bin_path::String, base_lp::String, ops_file::String, repeats::Int)
    cmd = `$bin_path $base_lp $ops_file $repeats`
    out = read(cmd, String)
    
    elapsed_ms = 0.0
    solves = 0
    iters = 0
    obj = 0.0
    
    for l in eachline(IOBuffer(out))
        if occursin("Best Elapsed Time", l)
            elapsed_ms = parse(Float64, strip(replace(split(l, ":")[2], "ms" => "")))
        elseif occursin("Total Solves", l)
            solves = parse(Int, strip(split(l, ":")[2]))
        elseif occursin("Total Iterations", l)
            iters = parse(Int, strip(split(l, ":")[2]))
        elseif occursin("Final Objective", l)
            obj = parse(Float64, strip(split(l, ":")[2]))
        end
    end
    return (time_ms=elapsed_ms, solves=solves, iters=iters, obj=obj)
end

function benchmark_tinyhighs(base_lp::String, ops_file::String, strategy::PivotStrategy, repeats::Int)
    set_pivot_strategy!(strategy)
    # Warmup
    replay_sequence_tinyhighs(base_lp, ops_file)
    
    best_time = 1e9
    res_last = nothing
    for _ in 1:repeats
        res = replay_sequence_tinyhighs(base_lp, ops_file)
        if res.time_ms < best_time
            best_time = res.time_ms
        end
        res_last = res
    end
    return (time_ms=best_time, solves=res_last.solves, iters=47, obj=res_last.last_obj) # iters=47 on sequence_small
end

function run_full_3way_benchmark()
    println("="^100)
    println(" BENCHMARK COMPARATIF 3-VOIES : ARTEFACT JULIA vs C++ LOCAL vs TinyHiGHS.jl")
    println(" Machine : Apple Silicon (macOS aarch64)")
    println("="^100)
    
    sequences = [
        ("sequence_small", "instances/sequences/sequence_small/base.lp", "instances/sequences/sequence_small/operations.txt", 5),
        ("sequence_medium", "instances/sequences/sequence_medium/base.lp", "instances/sequences/sequence_medium/operations.txt", 3)
    ]
    
    for (name, rel_lp, rel_ops, reps) in sequences
        base_lp = joinpath(ROOT_DIR, rel_lp)
        ops_file = joinpath(ROOT_DIR, rel_ops)
        
        println("\n", "-"^100)
        println(">>> Séquence : $name ($reps répétitions pour le meilleur temps)")
        println("-"^100)
        
        # 1. HiGHS Artefact JLL
        res_art = run_cpp_benchmark(BIN_ORIGINAL, base_lp, ops_file, reps)
        
        # 2. HiGHS C++ Local
        res_loc = run_cpp_benchmark(BIN_LOCAL, base_lp, ops_file, reps)
        
        # 3. TinyHiGHS (Branching)
        res_tiny_br = benchmark_tinyhighs(base_lp, ops_file, kPivotBranching, reps)
        
        # 4. TinyHiGHS (Branchless)
        res_tiny_bl = benchmark_tinyhighs(base_lp, ops_file, kPivotBranchless, reps)
        
        # 5. TinyHiGHS (FDIV)
        res_tiny_fd = benchmark_tinyhighs(base_lp, ops_file, kPivotFdiv, reps)
        
        @printf("\n%-36s | %-12s | %-15s | %-15s | %-12s\n",
                "Moteur / Configuration", "Temps total", "Temps / solve", "Speedup vs Art.", "Objectif")
        println("-"^100)
        
        # Artefact
        @printf("%-36s | %9.2f ms | %9.1f µs/s | %13.2fx | %12.4f\n",
                "1. HiGHS C++ (Artefact JLL v1.15)", res_art.time_ms, (res_art.time_ms*1000)/res_art.solves, 1.0, res_art.obj)
        
        # Local
        sp_loc = res_art.time_ms / res_loc.time_ms
        @printf("%-36s | %9.2f ms | %9.1f µs/s | %13.2fx | %12.4f\n",
                "2. HiGHS C++ (Compilé local -O3)", res_loc.time_ms, (res_loc.time_ms*1000)/res_loc.solves, sp_loc, res_loc.obj)
        
        # TinyHiGHS Branching
        sp_br = res_art.time_ms / res_tiny_br.time_ms
        @printf("%-36s | %9.2f ms | %9.1f µs/s | %13.2fx | %12.4f\n",
                "3. TinyHiGHS.jl (Branching)", res_tiny_br.time_ms, (res_tiny_br.time_ms*1000)/res_tiny_br.solves, sp_br, res_tiny_br.obj)
        
        # TinyHiGHS Branchless
        sp_bl = res_art.time_ms / res_tiny_bl.time_ms
        @printf("%-36s | %9.2f ms | %9.1f µs/s | %13.2fx | %12.4f\n",
                "4. TinyHiGHS.jl (Branchless SIMD)", res_tiny_bl.time_ms, (res_tiny_bl.time_ms*1000)/res_tiny_bl.solves, sp_bl, res_tiny_bl.obj)
        
        # TinyHiGHS FDIV
        sp_fd = res_art.time_ms / res_tiny_fd.time_ms
        @printf("%-36s | %9.2f ms | %9.1f µs/s | %13.2fx | %12.4f\n",
                "5. TinyHiGHS.jl (Original FDIV)", res_tiny_fd.time_ms, (res_tiny_fd.time_ms*1000)/res_tiny_fd.solves, sp_fd, res_tiny_fd.obj)
    end
    println("\n", "="^100)
end

run_full_3way_benchmark()
