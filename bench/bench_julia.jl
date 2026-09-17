# Référence de performance du port Julia (première mesure, plan §4.3) :
# build, rebuild, FTRAN/BTRAN, chaîne d'updates. Mêmes données que
# `hfactor_bench` (C++).
#
# Les appels sont faits derrière des barrières `@noinline` et leurs résultats
# sont consommés par `SINK` : sans cela LLVM élimine le travail et les temps
# s'effondrent (constaté : 1 ns/update).
#
#   julia --project=julia_simplex julia_simplex/bench/bench_julia.jl [n] [cas]
using TinyHiGHS
using Printf
using Statistics

include(joinpath(@__DIR__, "make_case.jl"))

const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
const CASE_PATH = length(ARGS) >= 2 ? ARGS[2] :
                  joinpath(tempdir(), "hfactor_bench_case_$(N).txt")

a_start, a_index, a_value = make_basis_case(N)
write_basis_case(CASE_PATH, N, a_start, a_index, a_value)
println("cas : n = ", N, " ; nnz = ", length(a_index), " ; ", CASE_PATH)

const BASE = collect(1:N)
const SINK = Ref(0.0)

@noinline function build_once!(n)
    f = HFactor(n, n, n, a_start, a_index, a_value, BASE)
    build!(f)
    SINK[] += f.l_start[end] + f.u_last_p[end]
    return f
end

@noinline function rebuild_once!(f)
    f.refactor_info.use = true
    build!(f)
    SINK[] += f.l_start[end]
    return nothing
end

@noinline function solve_pair!(f, v, s)
    for i ∈ eachindex(v.array)
        v.array[i] = ((i * 7 + s) % 13 - 6) * 0.5
    end
    v.count = -1
    ftranCall!(f, v, 1.0)
    SINK[] += v.array[1]
    for i ∈ eachindex(v.array)
        v.array[i] = ((i * 5 + s) % 11 - 5) * 0.25
    end
    v.count = -1
    btranCall!(f, v, 1.0)
    SINK[] += v.array[end]
    return nothing
end

@noinline function update_once!(f, iRow, aq, ep)
    update!(f, aq, ep, iRow)
    SINK[] += f.u_pivot_index[end]
    return nothing
end

function make_update(f::HFactor, t::Int)
    n = f.num_row
    iRow = (t - 1) % n + 1
    aq = HVector(n)
    aq.index[1] = iRow
    aq.count = 1
    for k ∈ 1:2
        r = (iRow * (k + 1) * 3) % n + 1
        (r == iRow || any(aq.index[i] == r for i ∈ 1:aq.count)) && continue
        aq.count += 1
        aq.index[aq.count] = r
        aq.array[r] = k == 1 ? 0.25 : -0.1
    end
    aq.array[iRow] = 1.5
    tight!(aq)
    aq.packFlag = true
    pack!(aq)
    ep = HVector(n)
    ep.index[1] = iRow
    ep.count = 1
    for k ∈ 1:2
        r = (iRow * (k + 2) * 5) % n + 1
        (r == iRow || any(ep.index[i] == r for i ∈ 1:ep.count)) && continue
        ep.count += 1
        ep.index[ep.count] = r
        ep.array[r] = k == 1 ? 0.5 : -0.75
    end
    tight!(ep)
    ep.packFlag = true
    pack!(ep)
    return iRow, aq, ep
end

function bench_build(repeats::Int)
    times = Float64[]
    for _ ∈ 1:repeats
        t0 = time_ns()
        build_once!(N)
        push!(times, (time_ns() - t0) / 1e6)
    end
    return times
end

function bench_solves(f::HFactor, repeats::Int, per_repeat::Int)
    v = HVector(f.num_row)
    times = Float64[]
    alloc = 0
    for r ∈ 1:repeats
        t0 = time_ns()
        for s ∈ 1:per_repeat
            solve_pair!(f, v, s)
        end
        push!(times, (time_ns() - t0) / 1e3 / (2 * per_repeat))
        alloc += @allocated solve_pair!(f, v, r)
    end
    return times, alloc
end

function bench_updates(repeats::Int, n_updates::Int)
    times = Float64[]
    for _ ∈ 1:repeats
        f = build_once!(N)
        updates = [make_update(f, t) for t ∈ 1:n_updates]
        t0 = time_ns()
        for (iRow, aq, ep) ∈ updates
            update_once!(f, iRow, aq, ep)
        end
        push!(times, (time_ns() - t0) / 1e3 / n_updates)
    end
    return times
end

function bench_rebuild(repeats::Int)
    times = Float64[]
    for _ ∈ 1:repeats
        f = build_once!(N)
        t0 = time_ns()
        rebuild_once!(f)
        push!(times, (time_ns() - t0) / 1e6)
    end
    return times
end

# Chauffe.
let f = build_once!(N)
    v = HVector(N)
    solve_pair!(f, v, 1)
    iRow, aq, ep = make_update(f, 1)
    update_once!(f, iRow, aq, ep)
end

build_ms = bench_build(5)
f = build_once!(N)
solve_us, solve_alloc = bench_solves(f, 3, 200)
update_us = bench_updates(3, 100)
rebuild_ms = bench_rebuild(3)

result = (; n=N, nnz=length(a_index), case_path=CASE_PATH, sink=SINK[],
    build_min_ms=minimum(build_ms), build_med_ms=median(build_ms),
    rebuild_min_ms=minimum(rebuild_ms), rebuild_med_ms=median(rebuild_ms),
    solve_min_us=minimum(solve_us), solve_med_us=median(solve_us),
    solve_alloc_per_call=solve_alloc ÷ 4,
    update_min_us=minimum(update_us), update_med_us=median(update_us))
@printf("build      : min %8.3f ms   median %8.3f ms\n",
    result.build_min_ms, result.build_med_ms)
@printf("rebuild    : min %8.3f ms   median %8.3f ms\n",
    result.rebuild_min_ms, result.rebuild_med_ms)
@printf("ftran+btran: min %8.3f us/appel  median %8.3f us/appel  (alloc %d o/appel)\n",
    result.solve_min_us, result.solve_med_us, result.solve_alloc_per_call)
@printf("update     : min %8.3f us/update median %8.3f us/update\n",
    result.update_min_us, result.update_med_us)
result
