# ==============================================================================
# Netlib Linear Programming Benchmark Suite
# Compares TinyHiGHS.jl against official HiGHS C++ and Netlib reference values
# ==============================================================================

using Printf
using TinyHiGHS

const NETLIB_DIR = normpath(joinpath(@__DIR__, "..", "instances", "netlib"))

# Reference optimal objective values from official Netlib documentation
const NETLIB_REFERENCE = Dict{String,Float64}(
    "afiro.lp"    => -4.6475314286e+02,
    "adlittle.lp" =>  2.2549496316e+05,
    "sc50a.lp"    => -6.4575077059e+01,
    "sc50b.lp"    => -7.0000000000e+01,
    "blend.lp"    => -3.0812149846e+01,
    "share2b.lp"  => -4.1573224074e+02,
)

function find_highs_binary()
    if haskey(ENV, "HIGHS_BIN") && isfile(ENV["HIGHS_BIN"])
        return ENV["HIGHS_BIN"]
    end
    for bin in ["highs", "/opt/homebrew/bin/highs", "/usr/local/bin/highs"]
        try
            p = run(pipeline(`which $bin`, devnull); wait=true)
            p.exitcode == 0 && return bin
        catch
        end
    end
    # Search in Julia artifacts (~/.julia/artifacts/*/bin/highs)
    artifacts_dir = expanduser("~/.julia/artifacts")
    if isdir(artifacts_dir)
        for (root, _, files) in walkdir(artifacts_dir)
            if "highs" in files
                cand = joinpath(root, "highs")
                if isfile(cand) && (filemode(cand) & 0o111 != 0)
                    return cand
                end
            end
        end
    end
    return nothing
end

function solve_with_highs_cli(highs_bin::String, lp_path::String)
    out = IOBuffer()
    err = IOBuffer()
    # Presolve off, simplex solver for fair comparison
    cmd = `$highs_bin --presolve off --solver simplex $lp_path`
    t0 = time_ns()
    run(pipeline(cmd, stdout=out, stderr=err))
    dt_ms = (time_ns() - t0) / 1e6

    output = String(take!(out))
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
    return (time_ms=dt_ms, obj=obj, iters=iters)
end

function solve_with_tinyhighs(lp_path::String)
    lp = read_lp(lp_path)
    engine = SimplexEngine(lp)

    # Warmup / precompilation
    initialise_for_solve!(engine)

    t0 = time_ns()
    status = solve!(engine)
    dt_ms = (time_ns() - t0) / 1e6

    obj = engine.info.primal_objective_value
    iters = engine.iteration_count
    return (status=status, time_ms=dt_ms, obj=obj, iters=iters,
            num_row=lp.num_row, num_col=lp.num_col, num_nz=length(lp.a_matrix.index))
end

function run_netlib_benchmarks()
    highs_bin = find_highs_binary()
    println("="^105)
    println(" NETLIB LINEAR PROGRAMMING BENCHMARK SUITE")
    println(" Comparing TinyHiGHS.jl vs Official HiGHS C++ and Netlib Reference Optima")
    println("="^105)
    println("HiGHS binary: ", isnothing(highs_bin) ? "Not found" : highs_bin)
    println()

    lp_files = filter(f -> endswith(f, ".lp"), sort(readdir(NETLIB_DIR)))
    if isempty(lp_files)
        println("No .lp files found in $NETLIB_DIR")
        return
    end

    @printf("%-12s | %-9s | %-6s | %-12s | %-12s | %-8s | %-16s | %-10s\n",
            "Instance", "Rows x Cols", "Nonzeros", "TinyHiGHS", "HiGHS C++", "Speedup", "Netlib Ref Obj", "Rel Err")
    println("-"^105)

    for f in lp_files
        lp_path = joinpath(NETLIB_DIR, f)
        res_tiny = solve_with_tinyhighs(lp_path)
        dim_str = "$(res_tiny.num_row)x$(res_tiny.num_col)"
        ref_obj = get(NETLIB_REFERENCE, f, NaN)
        rel_err = isfinite(ref_obj) ? abs(res_tiny.obj - ref_obj) / (1.0 + abs(ref_obj)) : NaN

        if !isnothing(highs_bin)
            res_highs = solve_with_highs_cli(highs_bin, lp_path)
            ratio = res_highs.time_ms / max(res_tiny.time_ms, 0.001)
            @printf("%-12s | %-11s | %8d | %8.2f ms | %8.2f ms | %7.2fx | %16.6e | %9.1e\n",
                    f, dim_str, res_tiny.num_nz, res_tiny.time_ms, res_highs.time_ms, ratio, ref_obj, rel_err)
        else
            @printf("%-12s | %-11s | %8d | %8.2f ms | %12s | %8s | %16.6e | %9.1e\n",
                    f, dim_str, res_tiny.num_nz, res_tiny.time_ms, "N/A", "-", ref_obj, rel_err)
        end
    end
    println("="^105)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_netlib_benchmarks()
end
