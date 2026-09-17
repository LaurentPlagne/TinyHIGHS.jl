# Génère une matrice de base creuse à dominance diagonale, dans le format des
# cas de l'oracle HFactor (num_row num_col num_basic nnz, a_start/a_index
# 0-based, a_value %a, basic_index 0-based). Déterministe, sans RNG.
using Printf

hexof(x::Float64) = @sprintf("%a", x)

function make_basis_case(n::Int; nnz_per_col::Int=3)
    a_start = Int[1]
    a_index = Int[]
    a_value = Float64[]
    for j ∈ 1:n
        entries = Tuple{Int,Float64}[(j, 4.0 + (j % 7) * 0.1)]
        for k ∈ 1:(nnz_per_col - 1)
            r = (j * (k + 1) * 3) % n + 1
            any(e -> e[1] == r, entries) && continue
            push!(entries, (r, k == 1 ? 0.1 : -0.05))
        end
        sort!(entries; by=first)
        for (r, v) ∈ entries
            push!(a_index, r)
            push!(a_value, v)
        end
        push!(a_start, length(a_index) + 1)
    end
    return a_start, a_index, a_value
end

function write_basis_case(path::AbstractString, n::Int, a_start, a_index, a_value)
    open(path, "w") do io
        println(io, n, " ", n, " ", n, " ", length(a_index))
        println(io, join(a_start .- 1, " "))
        println(io, join(a_index .- 1, " "))
        println(io, join(hexof.(a_value), " "))
        println(io, join(0:(n - 1), " "))
    end
    return path
end
