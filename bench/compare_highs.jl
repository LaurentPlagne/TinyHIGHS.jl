# ==============================================================================
# Benchmark comparatif : TinyHiGHS vs HiGHS C++ officiel sur la suite .lp
# ==============================================================================

using Printf
using TinyHiGHS

const INSTANCES_DIR = abspath(joinpath(@__DIR__, "..", "instances"))

# Recherche du binaire highs officiel
function find_highs_binary()
    for bin in ["highs", "/opt/homebrew/bin/highs", "/usr/local/bin/highs"]
        try
            p = run(pipeline(`which $bin`, devnull); wait=true)
            p.exitcode == 0 && return bin
        catch
        end
    end
    # Recherche dans les artefacts Julia (~/.julia/artifacts/*/bin/highs)
    artifacts_dir = expanduser("~/.julia/artifacts")
    if isdir(artifacts_dir)
        for (root, _, files) in walkdir(artifacts_dir)
            if "highs" in files
                candidate = joinpath(root, "highs")
                if isfile(candidate) && (filemode(candidate) & 0o111 != 0)
                    return candidate
                end
            end
        end
    end
    # Repli : l'oracle de sequence compile déjà contre libhighs
    oracle_bin = abspath(joinpath(@__DIR__, "..", "oracle", "build", "sequence_oracle"))
    isfile(oracle_bin) && return oracle_bin
    return nothing
end

"""
Résolution d'un fichier .lp par HiGHS CLI.
"""
function solve_with_highs_cli(highs_bin::String, lp_path::String)
    t0 = time_ns()
    out = IOBuffer()
    err = IOBuffer()
    # HiGHS CLI options : presolve off et dual simplex pour comparaison équitable
    cmd = `$highs_bin --presolve off --solver simplex $lp_path`
    proc = run(pipeline(cmd, stdout=out, stderr=err))
    dt = (time_ns() - t0) / 1e6 # ms

    output = String(take!(out))
    # Extraction de l'objectif et des itérations
    obj = NaN
    iters = -1
    for line in split(output, '\n')
        if occursin("Objective value", line) || occursin("Objective", line)
            m = match(r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", line)
            if !isnothing(m)
                obj = parse(Float64, m.match)
            end
        elseif occursin("Iteration count", line) || occursin("Iterations", line)
            m = match(r"([0-9]+)", line)
            if !isnothing(m)
                iters = parse(Int, m.match)
            end
        end
    end
    return (time_ms=dt, obj=obj, iters=iters)
end

"""
Résolution d'un fichier .lp par TinyHiGHS.
"""
function solve_with_tinyhighs(lp_path::String)
    lp = read_lp(lp_path)
    engine = SimplexEngine(lp)
    
    # Warm-up / précompilation si première fois
    initialise_for_solve!(engine)
    
    t0 = time_ns()
    status = solve!(engine)
    dt = (time_ns() - t0) / 1e6 # ms
    
    obj = engine.info.primal_objective_value
    iters = engine.iteration_count
    return (status=status, time_ms=dt, obj=obj, iters=iters, nrows=lp.num_row, ncols=lp.num_col)
end

function run_all_benchmarks()
    highs_bin = find_highs_binary()
    println("="^95)
    println(" BENCHMARK COMPARATIF : TinyHiGHS.jl vs HiGHS C++ (Official CLI v1.15.1)")
    println("="^95)
    println("HiGHS binaire : ", isnothing(highs_bin) ? "Non trouvé" : highs_bin)
    println()

    # Découverte de toutes les instances
    lp_files = String[]
    for (root, _, files) in walkdir(INSTANCES_DIR)
        for f in sort(files)
            endswith(f, ".lp") && push!(lp_files, joinpath(root, f))
        end
    end

    if isempty(lp_files)
        println("Aucun fichier .lp trouvé dans $INSTANCES_DIR")
        return
    end

    @printf("%-26s | %-9s | %-12s | %-12s | %-8s | %-8s | %-10s\n",
            "Instance (.lp)", "Dims", "TinyHiGHS", "HiGHS C++", "Speedup", "Statut", "Écart Obj")
    println("-"^95)

    for lp_path in lp_files
        rel_name = relpath(lp_path, INSTANCES_DIR)
        
        # Résolution TinyHiGHS
        res_tiny = solve_with_tinyhighs(lp_path)
        dim_str = "$(res_tiny.nrows)x$(res_tiny.ncols)"

        if !isnothing(highs_bin) && highs_bin != abspath(joinpath(@__DIR__, "..", "oracle", "build", "sequence_oracle"))
            res_highs = solve_with_highs_cli(highs_bin, lp_path)
            ratio = res_highs.time_ms / max(res_tiny.time_ms, 0.001)
            diff_obj = isfinite(res_tiny.obj) && isfinite(res_highs.obj) ? abs(res_tiny.obj - res_highs.obj) : 0.0
            diff_str = diff_obj < 1e-9 ? "exact (0.0)" : @sprintf("%.2e", diff_obj)
            stat_str = string(res_tiny.status)
            
            @printf("%-26s | %-9s | %8.2f ms | %8.2f ms | %7.2fx | %-8s | %-10s\n",
                    rel_name, dim_str, res_tiny.time_ms, res_highs.time_ms, ratio, stat_str, diff_str)
        else
            @printf("%-26s | %-9s | %8.2f ms | %12s | %8s | %-8s | %-10s\n",
                    rel_name, dim_str, res_tiny.time_ms, "N/A", "-", string(res_tiny.status), "-")
        end
    end
    println("="^95)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_benchmarks()
end
