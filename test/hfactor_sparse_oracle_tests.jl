# Oracle C++ — solves FTRAN/BTRAN à liste d'indices FOURNIE et chaînes
# d'updates : complète l'oracle HFactor dense (`count = -1`) sur l'ordre de
# `index`, que les cas denses n'exercent pas, et sur les chaînes longues.
using Printf

const HFACTOR_SPARSE_BIN = abspath(joinpath(@__DIR__, "..", "oracle", "build",
    "hfactor_sparse_oracle"))

"""État packé d'un `HVector`, dans l'ordre du lecteur C++ (0-based, %a)."""
function write_sparse_packed(io::IO, v::HVector)
    println(io, v.size, " ", v.count)
    println(io, join((v.index[i] - 1 for i ∈ 1:max(v.count, 0)), " "))
    println(io, join(hexof.(v.array), " "))
    println(io, v.packCount)
    println(io, join((v.packIndex[i] - 1 for i ∈ 1:v.packCount), " "))
    println(io, join((hexof(v.packValue[i]) for i ∈ 1:v.packCount), " "))
    return nothing
end

function write_sparse_case(io::IO, m::SparseMatrix, basic, updates, rhs_list)
    println(io, m.num_row, " ", m.num_col, " ", length(m.index))
    println(io, join(m.start .- 1, " "))
    println(io, join(m.index .- 1, " "))
    println(io, join(hexof.(m.value), " "))
    println(io, join(basic .- 1, " "))
    println(io, length(updates))
    for (iRow, aq, ep) ∈ updates
        println(io, iRow - 1)
        write_sparse_packed(io, aq)
        write_sparse_packed(io, ep)
    end
    println(io, length(rhs_list))
    for (idx, vals, dens) ∈ rhs_list
        println(io, length(idx), " ", hexof(dens))
        for (i, v) ∈ zip(idx, vals)
            println(io, i - 1, " ", hexof(v))
        end
        for v ∈ vals
            println(io, hexof(v))
        end
    end
    return nothing
end

"""Cas : matrice bien conditionnée, base structurelle, updates et solves creux."""
function sparse_solve_case(rng::AbstractRNG; n::Int=8, n_updates::Int=0, n_rhs::Int=3)
    A = zeros(n, n)
    rows = shuffle(rng, 1:n)
    for j ∈ 1:n
        A[rows[j], j] = (rand(rng) < 0.5 ? 1.0 : -1.0) * (2 + rand(rng))
        for _ ∈ 1:rand(rng, 0:2)
            A[rand(rng, 1:n), j] += randn(rng)
        end
    end
    updates = Tuple{Int,HVector,HVector}[]
    for _ ∈ 1:n_updates
        iRow = rand(rng, 1:n)
        aq = HVector(n)
        aq.count = 1
        aq.index[1] = iRow
        aq.array[iRow] = 1.0 + rand(rng)
        for _ ∈ 1:rand(rng, 0:2)
            r = rand(rng, 1:n)
            r == iRow && continue
            aq.count += 1
            aq.index[aq.count] = r
            aq.array[r] = randn(rng)
        end
        tight!(aq)
        aq.packFlag = true
        pack!(aq)
        ep = HVector(n)
        ep.count = rand(rng, 1:min(3, n))
        for k ∈ 1:ep.count
            r = rand(rng, 1:n)
            ep.index[k] = r
            ep.array[r] = randn(rng)
        end
        tight!(ep)
        ep.packFlag = true
        pack!(ep)
        push!(updates, (iRow, aq, ep))
    end
    rhs_list = Vector{Tuple{Vector{Int},Vector{Float64},Float64}}()
    for _ ∈ 1:n_rhs
        count = rand(rng, 1:min(4, n))
        push!(rhs_list, (sort(shuffle(rng, 1:n)[1:count]), randn(rng, count),
            rand(rng, (0.01, 0.1, 0.5, 1.0))))
    end
    return A, collect(1:n), updates, rhs_list
end

"""Lignes attendues du port : `(count, index, array %a)` pour FTRAN puis BTRAN."""
function port_sparse_case(A, basic, updates, rhs_list)
    n = size(A, 1)
    m = csc_from_dense(A)
    f = HFactor(size(A, 2), n, n, m.start, m.index, m.value, copy(basic))
    build!(f) == 0 || error("base singulière")
    for (iRow, aq, ep) ∈ updates
        update!(f, aq, ep, iRow)
    end
    lines = String[]
    for (idx, vals, dens) ∈ rhs_list
        v = HVector(n)
        v.count = length(idx)
        v.index[1:length(idx)] = idx
        v.array[idx] .= vals
        ftranCall!(f, v, dens)
        push!(lines, join(vcat(string(v.count),
            string.(v.index[1:v.count] .- 1), hexof.(v.array)), " "))
        w = HVector(n)
        w.count = length(idx)
        w.index[1:length(idx)] = idx
        w.array[idx] .= vals
        btranCall!(f, w, dens)
        push!(lines, join(vcat(string(w.count),
            string.(w.index[1:w.count] .- 1), hexof.(w.array)), " "))
    end
    return lines
end

if !isfile(HFACTOR_SPARSE_BIN)
    @info "oracle HFactor creux absent — tests ignorés (nécessite oracle/build.sh)"
else
@testset "oracle C++ — HFactor creux (index fournis, chaînes d'updates)" begin
    old_strat = ACTIVE_PIVOT_STRATEGY[]
    set_pivot_strategy!(kPivotBranching)
    try
        if isfile(HFACTOR_SPARSE_BIN)
            rng = MersenneTwister(20260930)
            problems = String[]
            for (icase, (n, n_updates)) ∈ enumerate(vcat([(rand(rng, 4:12), 0) for _ ∈ 1:12],
                [(rand(rng, 4:12), 40) for _ ∈ 1:8]))
                A, basic, updates, rhs_list = sparse_solve_case(rng; n=n,
                    n_updates=n_updates, n_rhs=3)
                input = IOBuffer()
                write_sparse_case(input, csc_from_dense(A), basic, updates,
                    rhs_list)
                got = split(read(pipeline(`$HFACTOR_SPARSE_BIN`;
                    stdin=IOBuffer(String(take!(input)))), String), '\n')
                expected = port_sparse_case(A, basic, updates, rhs_list)
                for (k, line) ∈ enumerate(expected)
                    strip(got[k]) == line ||
                        push!(problems, "cas $icase ligne $k : $line ≠ $(strip(got[k]))")
                    length(problems) > 5 && break
                end
                length(problems) > 5 && break
            end
            @test isempty(problems)
            isempty(problems) || @info "écarts solves creux" problems
        end
    finally
        set_pivot_strategy!(old_strat)
    end
end
end
