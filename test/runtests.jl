using Test
using Random
using TinyHiGHS

"""Matrice CSC 1-based depuis une matrice dense (petits cas de test)."""
function csc_from_dense(A::Matrix{Float64})
    num_row, num_col = size(A)
    a_start = Vector{Int}(undef, num_col + 1)
    a_index = Int[]
    a_value = Float64[]
    a_start[1] = 1
    for j ∈ 1:num_col
        for i ∈ 1:num_row
            A[i, j] == 0.0 && continue
            push!(a_index, i)
            push!(a_value, A[i, j])
        end
        a_start[j + 1] = length(a_index) + 1
    end
    return SparseMatrix(num_col, num_row, a_start, a_index, a_value)
end

include("lp_io_tests.jl")
include("hvector_tests.jl")
include("types_tests.jl")
include("oracle_tests.jl")
include("hfactor_oracle_tests.jl")
include("hfactor_pivot_strategy_tests.jl")
include("hfactor_sparse_oracle_tests.jl")
include("sparse_matrix_tests.jl")
include("matrix_oracle_tests.jl")
include("random_oracle_tests.jl")
include("hash_oracle_tests.jl")
include("nla_tests.jl")
include("engine_tests.jl")
include("dual_oracle_tests.jl")
include("primal_oracle_tests.jl")
include("sequence_oracle_tests.jl")
include("moi_tests.jl")
include("netlib_tests.jl")
