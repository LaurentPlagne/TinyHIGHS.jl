# ==============================================================================
# Benchmark comparatif :
#   1. HiGHS_artifact (artefact Julia officiel HiGHS_jll v1.15.1)
#   2. HiGHS_branchless (build local de la PR perf/simd-branchless-pivots)
#   3. TinyHiGHS (Julia natif zéro-allocation, stratégies de pivot)
# ==============================================================================

using Printf
using TinyHiGHS

const ROOT_DIR = dirname(@__DIR__)
const BIN_ORIGINAL = joinpath(ROOT_DIR, "contrib_highs", "cpp", "replay_sequence_original")
const BIN_LOCAL = joinpath(ROOT_DIR, "contrib_highs", "cpp", "replay_sequence")

include(joinpath(ROOT_DIR, "bench", "compare_sequences.jl"))

function run_cpp_benchmark(bin_path::String, base_lp::String, ops_file::String, repeats::Int)
    # Keep the Julia benchmark usable when no C++ runner has been built.
    isfile(bin_path) && Sys.which(bin_path) !== nothing || return nothing
    cmd = `$bin_path $base_lp $ops_file $repeats`
    out = try
        read(cmd, String)
    catch err
        @warn "C++ benchmark runner failed; skipping this comparison" bin_path exception=(err, catch_backtrace())
        return nothing
    end
    
    elapsed_ms = 0.0
    solves = 0
    iters = 0
    obj = 0.0
    status = nothing
    
    for l in eachline(IOBuffer(out))
        if occursin("Best Elapsed Time", l)
            elapsed_ms = parse(Float64, strip(replace(split(l, ":")[2], "ms" => "")))
        elseif occursin("Total Solves", l)
            solves = parse(Int, strip(split(l, ":")[2]))
        elseif occursin("Total Iterations", l)
            iters = parse(Int, strip(split(l, ":")[2]))
        elseif occursin("Final Objective", l)
            obj = parse(Float64, strip(split(l, ":")[2]))
        elseif occursin("Final Status Code", l)
            status = parse(Int, strip(split(l, ":")[2]))
        end
    end
    solves > 0 || return nothing
    return (time_ms=elapsed_ms, solves=solves, iters=iters, obj=obj,
        status=status)
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
    return (time_ms=best_time, solves=res_last.solves, iters=res_last.iters,
        obj=res_last.last_obj, status=res_last.status)
end

function _print_unavailable(label::AbstractString)
    @printf("%-36s | %12s | %15s | %15s | %10s | %12s | %s\n",
            label, "N/A", "N/A", "N/A", "N/A", "N/A", "runner indisponible")
end

function _print_benchmark_row(label::AbstractString, result, reference)
    result === nothing && return _print_unavailable(label)
    speedup = reference === nothing ? NaN : reference.time_ms / max(result.time_ms, eps())
    result_status_optimal = !hasproperty(result, :status) ||
        result.status === nothing || result.status == TinyHiGHS.kOptimal ||
        result.status == Int(TinyHiGHS.kOptimal)
    objective_status = !result_status_optimal ?
        "NONOPT" : reference === nothing ? "n/a" :
        (isapprox(result.obj, reference.obj; rtol=1e-8, atol=1e-8) ? "OK" : "DIFF")
    speedup_text = reference === nothing ? "N/A" : @sprintf("%12.2fx", speedup)
    @printf("%-36s | %9.2f ms | %9.1f µs/s | %10d | %12s | %12.4f | %s\n",
            label, result.time_ms, (result.time_ms * 1000) / max(result.solves, 1),
            result.iters, speedup_text, result.obj, objective_status)
end

function _git_revision()
    try
        return readchomp(`git -C $ROOT_DIR rev-parse --short HEAD`)
    catch
        return "unknown"
    end
end

function _print_benchmark_metadata()
    compiler = Sys.which("clang++")
    println(" Julia version : ", VERSION)
    println(" CPU           : ", Sys.CPU_NAME)
    println(" Machine       : ", Sys.MACHINE, " (", Sys.KERNEL, ")")
    println(" Word size     : ", Sys.WORD_SIZE)
    println(" Git revision  : ", _git_revision())
    println(" C++ compiler  : ", compiler === nothing ? "not found" : compiler)
    println(" C++ runners   : ", BIN_ORIGINAL, " ; ", BIN_LOCAL)
end

function run_full_3way_benchmark()
    println("="^100)
    println(" BENCHMARK COMPARATIF : HiGHS_artifact vs HiGHS_branchless vs TinyHiGHS")
    _print_benchmark_metadata()
    println(" Baseline C++ non patchée : HiGHS_artifact; le build local est HiGHS_branchless.")
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
        
        # 1. HiGHS_artifact: official Julia artifact, used as the baseline.
        res_art = run_cpp_benchmark(BIN_ORIGINAL, base_lp, ops_file, reps)
        
        # 2. HiGHS_branchless: local C++ build selected by run_bench_cpp.sh.
        res_loc = run_cpp_benchmark(BIN_LOCAL, base_lp, ops_file, reps)
        
        # 3. TinyHiGHS (Branching)
        res_tiny_br = benchmark_tinyhighs(base_lp, ops_file, kPivotBranching, reps)
        
        # 4. TinyHiGHS (Branchless)
        res_tiny_bl = benchmark_tinyhighs(base_lp, ops_file, kPivotBranchless, reps)
        
        # 5. TinyHiGHS (FDIV)
        res_tiny_fd = benchmark_tinyhighs(base_lp, ops_file, kPivotFdiv, reps)
        
        @printf("\n%-36s | %-12s | %-15s | %-10s | %-12s | %-12s | %-8s\n",
                "Moteur / Configuration", "Temps total", "Temps / solve", "Itérations", "Speedup vs Art.", "Objectif", "Statut")
        println("-"^120)

        _print_benchmark_row("1. HiGHS_artifact (official v1.15)", res_art, nothing)
        _print_benchmark_row("2. HiGHS_branchless (local PR, -O3)", res_loc, res_art)
        _print_benchmark_row("3. TinyHiGHS (branching)", res_tiny_br, res_art)
        _print_benchmark_row("4. TinyHiGHS_branchless (SIMD)", res_tiny_bl, res_art)
        _print_benchmark_row("5. TinyHiGHS (FDIV reference)", res_tiny_fd, res_art)

        if res_art !== nothing
            for (label, result) in (("Branching", res_tiny_br), ("Branchless SIMD", res_tiny_bl), ("Original FDIV", res_tiny_fd))
                result !== nothing && result.status === TinyHiGHS.kOptimal &&
                    (res_art.status === nothing || res_art.status == TinyHiGHS.kOptimal ||
                     res_art.status == Int(TinyHiGHS.kOptimal)) &&
                    !isapprox(result.obj, res_art.obj; rtol=1e-8, atol=1e-8) &&
                    @warn "Objective mismatch in 3-way benchmark" sequence=name strategy=label cpp=res_art.obj tiny=result.obj
            end
        end
    end
    println("\n", "="^100)
end

run_full_3way_benchmark()
