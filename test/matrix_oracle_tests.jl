# Oracle C++ de `HighsSparseMatrix` : comparaison bit-à-bit du port M2a avec la
# source INCHANGÉE. Voir docs/architecture/portage-julia-simplexe-highs.md §4.1.

const MATRIX_ORACLE_BIN = abspath(joinpath(@__DIR__, "..", "oracle", "build",
    "matrix_oracle"))

"""Cas : matrice CSC 1-based + entrées des requêtes."""
struct MatrixCase
    num_col::Int
    num_row::Int
    a_start::Vector{Int}
    a_index::Vector{Int}
    a_value::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    col_dense::HVector
    col_sparse::HVector
    expected_density::Float64
    from_index::Int
    switch_density::Float64
    dot_array::Vector{Float64}
    use_col_dot::Int
    collect_state::HVector
    use_col_collect::Int
    multiplier::Float64
    in_partition::Vector{Bool}
    partition_updates::Vector{Tuple{Int,Int}}
    col_scale::Vector{Float64}
    row_scale::Vector{Float64}
    scale_col_idx::Int
    scale_col_val::Float64
    scale_row_idx::Int
    scale_row_val::Float64
end

function write_sparse_state(io::IO, v::HVector)
    println(io, v.size, " ", v.count)
    println(io, join((v.index[i] - 1 for i ∈ 1:max(v.count, 0)), " "))
    println(io, join(hexof.(v.array), " "))
    return nothing
end

function write_matrix_case(io::IO, c::MatrixCase)
    println(io, c.num_row, " ", c.num_col, " ", length(c.a_index))
    println(io, join(c.a_start .- 1, " "))
    println(io, join(c.a_index .- 1, " "))
    println(io, join(hexof.(c.a_value), " "))
    println(io, join(hexof.(c.x), " "))
    println(io, join(hexof.(c.y), " "))
    # Ordre du lecteur C++ : col_dense, dot, collect, puis col_sparse + seuils.
    write_sparse_state(io, c.col_dense)
    println(io, join(hexof.(c.dot_array), " "))
    println(io, c.use_col_dot - 1)
    write_sparse_state(io, c.collect_state)
    println(io, c.use_col_collect - 1, " ", hexof(c.multiplier))
    write_sparse_state(io, c.col_sparse)
    println(io, hexof(c.expected_density), " ", c.from_index - 1, " ",
        hexof(c.switch_density))
    println(io, join(Int.(c.in_partition), " "))
    println(io, length(c.partition_updates))
    for (var_in, var_out) ∈ c.partition_updates
        println(io, var_in - 1, " ", var_out - 1)
    end
    println(io, join(hexof.(c.col_scale), " "))
    println(io, join(hexof.(c.row_scale), " "))
    println(io, c.scale_col_idx - 1, " ", hexof(c.scale_col_val))
    println(io, c.scale_row_idx - 1, " ", hexof(c.scale_row_val))
    return nothing
end

function run_matrix_oracle(input::String)
    out = IOBuffer()
    err = IOBuffer()
    process = run(pipeline(`$MATRIX_ORACLE_BIN`; stdin=IOBuffer(input),
        stdout=out, stderr=err))
    success(process) ||
        error("oracle matrice : code $(process.exitcode) ; $(String(take!(err)))")
    return String(take!(out))
end

function random_matrix_case(rng::AbstractRNG)
    num_row = rand(rng, 3:8)
    num_col = rand(rng, 3:8)
    a_start = Int[1]
    a_index = Int[]
    a_value = Float64[]
    for j ∈ 1:num_col
        rows = sort(shuffle(rng, 1:num_row)[1:rand(rng, 1:3)])
        for r ∈ rows
            push!(a_index, r)
            push!(a_value, randn(rng))
        end
        push!(a_start, length(a_index) + 1)
    end
    x = randn(rng, num_col)
    y = randn(rng, num_row)
    col_dense = HVector(num_row)
    col_dense.count = -1
    col_dense.array .= randn(rng, num_row)
    col_sparse = HVector(num_row)
    rows = sort(shuffle(rng, 1:num_row)[1:rand(rng, 1:num_row)])
    for (k, r) ∈ enumerate(rows)
        col_sparse.index[k] = r
        col_sparse.array[r] = rand(rng, Bool) ? randn(rng) : 0.0
    end
    col_sparse.count = length(rows)
    collect_state = HVector(num_row)
    c_rows = sort(shuffle(rng, 1:num_row)[1:rand(rng, 1:num_row)])
    for (k, r) ∈ enumerate(c_rows)
        collect_state.index[k] = r
        collect_state.array[r] = randn(rng)
    end
    collect_state.count = length(c_rows)
    from_index = rand(rng, 1:max(col_sparse.count, 1))
    in_partition = rand(rng, Bool, num_col)
    # Contrat de `update` : `var_in` est dans la partition (il en sort),
    # `var_out` n'y est pas (il y entre). Les logiques n'ont pas d'entrées.
    member = copy(in_partition)
    partition_updates = Tuple{Int,Int}[]
    for _ ∈ 1:rand(rng, 0:3)
        ins = [j for j ∈ 1:num_col if member[j]]
        outs = [j for j ∈ 1:num_col if !member[j]]
        (isempty(ins) && isempty(outs)) && break
        var_in = isempty(ins) ? rand(rng, (num_col + 1):(num_col + num_row)) :
                 rand(rng, ins)
        var_out = isempty(outs) ? rand(rng, (num_col + 1):(num_col + num_row)) :
                  rand(rng, outs)
        var_in <= num_col && (member[var_in] = false)
        var_out <= num_col && (member[var_out] = true)
        push!(partition_updates, (var_in, var_out))
    end
    return MatrixCase(num_col, num_row, a_start, a_index, a_value, x, y,
        col_dense, col_sparse, rand(rng) < 0.5 ? 0.0 : 1e-9, from_index,
        rand(rng, Bool) ? 10.0 : 0.0, randn(rng, num_row),
        rand(rng, 1:(num_col + num_row)), collect_state,
        rand(rng, 1:(num_col + num_row)), randn(rng), in_partition,
        partition_updates, exp.(randn(rng, num_col) * 0.5),
        exp.(randn(rng, num_row) * 0.5), rand(rng, 1:num_col),
        exp(randn(rng) * 0.5), rand(rng, 1:num_row), exp(randn(rng) * 0.5))
end

"""
Cas ciblé pour la somme quad : deux rangées annulent exactement une colonne du
paquet (`add!` + `cleanup!`), ce qu'un tirage uniforme n'atteint jamais.
`row_ep = e1 + e2`, `A[:, 1] = e1 - e2`, `A[:, 2] = 2 e1`, `A[:, 3] = e2` :
la colonne 1 se somme à `floatmin` puis est retirée, les colonnes 2 et 3
restent ; l'ordre `nonzeroinds` après échange (`[3, 2]`) est celui de la
source. `dense` force la branche dense (annulation en `HighsCDouble`).
"""
function cleanup_matrix_case(dense::Bool)
    num_col, num_row = 3, 2
    a_start = [1, 3, 4, 5]
    a_index = [1, 2, 1, 2]
    a_value = [1.0, -1.0, 2.0, 1.0]
    x = [1.0, -2.0, 3.0]
    y = [1.0, -1.0]
    col_dense = HVector(num_row)
    col_dense.array[1] = 1.0
    col_dense.array[2] = 1.0
    col_dense.count = 2
    col_dense.index[1] = 1
    col_dense.index[2] = 2
    col_sparse = HVector(num_row)
    col_sparse.array[1] = 1.0
    col_sparse.array[2] = 1.0
    col_sparse.count = 2
    col_sparse.index[1] = 1
    col_sparse.index[2] = 2
    collect_state = HVector(num_row)
    collect_state.count = 0
    return MatrixCase(num_col, num_row, a_start, a_index, a_value, x, y,
        col_dense, col_sparse, dense ? 1.0 : 0.0, 1, Inf, [0.5, -0.5],
        num_col + num_row, collect_state, 1, 1.0, [true, true, true],
        Tuple{Int,Int}[], [1.0, 1.0, 1.0], [1.0, 1.0], 1, 1.0, 1, 1.0)
end

"""Journal du port pour un cas, dans l'ordre exact de l'oracle."""
function port_matrix_case(c::MatrixCase)
    m = SparseMatrix(c.num_col, c.num_row, c.a_start, c.a_index, c.a_value)
    out = String[]
    prod = zeros(c.num_row)
    product!(prod, m, c.x)
    push!(out, string(length(prod)))
    append!(out, hexof.(prod))
    prod_t = zeros(c.num_col)
    product_transpose!(prod_t, m, c.y)
    push!(out, string(length(prod_t)))
    append!(out, hexof.(prod_t))
    result_pc = HVector(c.num_col)
    price_by_column!(m, result_pc, c.col_dense)
    push!(out, string(result_pc.count))
    append!(out, string.(result_pc.index[1:result_pc.count] .- 1))
    append!(out, hexof.(result_pc.array))
    result_pcq = HVector(c.num_col)
    price_by_column!(m, result_pcq, c.col_dense, true)
    push!(out, string(result_pcq.count))
    append!(out, string.(result_pcq.index[1:result_pcq.count] .- 1))
    append!(out, hexof.(result_pcq.array))
    push!(out, hexof(compute_dot(m, c.dot_array, c.use_col_dot)))
    collect_aj!(m, c.collect_state, c.use_col_collect, c.multiplier)
    push!(out, string(c.collect_state.count))
    append!(out, string.(c.collect_state.index[1:c.collect_state.count] .- 1))
    append!(out, hexof.(c.collect_state.array))
    ensure_rowwise!(m)
    push!(out, string(length(m.start)))
    append!(out, string.(m.start .- 1))
    push!(out, string(length(m.index)))
    append!(out, string.(m.index .- 1))
    push!(out, string(length(m.value)))
    append!(out, hexof.(m.value))
    prod_r = zeros(c.num_row)
    product!(prod_r, m, c.x)
    push!(out, string(length(prod_r)))
    append!(out, hexof.(prod_r))
    prod_tr = zeros(c.num_col)
    product_transpose!(prod_tr, m, c.y)
    push!(out, string(length(prod_tr)))
    append!(out, hexof.(prod_tr))
    result_pr = HVector(c.num_col)
    price_by_row!(m, result_pr, c.col_sparse)
    push!(out, string(result_pr.count))
    append!(out, string.(result_pr.index[1:result_pr.count] .- 1))
    append!(out, hexof.(result_pr.array))
    result_pw = HVector(c.num_col)
    price_by_row_with_switch!(m, result_pw, c.col_sparse, c.expected_density,
        c.from_index, c.switch_density)
    push!(out, string(result_pw.count))
    append!(out, string.(result_pw.index[1:result_pw.count] .- 1))
    append!(out, hexof.(result_pw.array))
    result_prq = HVector(c.num_col)
    price_by_row!(m, result_prq, c.col_sparse, true)
    push!(out, string(result_prq.count))
    append!(out, string.(result_prq.index[1:result_prq.count] .- 1))
    append!(out, hexof.(result_prq.array))
    result_pwq = HVector(c.num_col)
    price_by_row_with_switch!(m, result_pwq, c.col_sparse, c.expected_density,
        c.from_index, c.switch_density, true)
    push!(out, string(result_pwq.count))
    append!(out, string.(result_pwq.index[1:result_pwq.count] .- 1))
    append!(out, hexof.(result_pwq.array))

    # Pricing partitionné.
    m_col = SparseMatrix(c.num_col, c.num_row, c.a_start, c.a_index, c.a_value)
    part = SparseMatrix(c.num_col, c.num_row)
    create_rowwise_partitioned!(part, m_col, c.in_partition)
    for v ∈ (part.start, part.p_end, part.index)
        push!(out, string(length(v)))
        append!(out, string.(v .- 1))
    end
    push!(out, string(length(part.value)))
    append!(out, hexof.(part.value))
    result_pp = HVector(c.num_col)
    price_by_row_with_switch!(part, result_pp, c.col_sparse,
        c.expected_density, c.from_index, c.switch_density)
    push!(out, string(result_pp.count))
    append!(out, string.(result_pp.index[1:result_pp.count] .- 1))
    append!(out, hexof.(result_pp.array))
    result_ppq = HVector(c.num_col)
    price_by_row_with_switch!(part, result_ppq, c.col_sparse,
        c.expected_density, c.from_index, c.switch_density, true)
    push!(out, string(result_ppq.count))
    append!(out, string.(result_ppq.index[1:result_ppq.count] .- 1))
    append!(out, hexof.(result_ppq.array))
    for (var_in, var_out) ∈ c.partition_updates
        update!(part, var_in, var_out, m_col)
    end
    if !isempty(c.partition_updates)
        for v ∈ (part.start, part.p_end, part.index)
            push!(out, string(length(v)))
            append!(out, string.(v .- 1))
        end
        push!(out, string(length(part.value)))
        append!(out, hexof.(part.value))
        result_pu = HVector(c.num_col)
        price_by_row_with_switch!(part, result_pu, c.col_sparse,
            c.expected_density, c.from_index, c.switch_density)
        push!(out, string(result_pu.count))
        append!(out, string.(result_pu.index[1:result_pu.count] .- 1))
        append!(out, hexof.(result_pu.array))
        result_puq = HVector(c.num_col)
        price_by_row_with_switch!(part, result_puq, c.col_sparse,
            c.expected_density, c.from_index, c.switch_density, true)
        push!(out, string(result_puq.count))
        append!(out, string.(result_puq.index[1:result_puq.count] .- 1))
        append!(out, hexof.(result_puq.array))
    end

    # Échelles : applique/désapplique, puis scale_col/scale_row.
    scale = Scale(c.col_scale, c.row_scale)
    apply_scale!(m_col, scale)
    push!(out, string(length(m_col.value)))
    append!(out, hexof.(m_col.value))
    unapply_scale!(m_col, scale)
    push!(out, string(length(m_col.value)))
    append!(out, hexof.(m_col.value))
    scale_col!(m_col, c.scale_col_idx, c.scale_col_val)
    push!(out, string(length(m_col.value)))
    append!(out, hexof.(m_col.value))
    scale_row!(m_col, c.scale_row_idx, c.scale_row_val)
    push!(out, string(length(m_col.value)))
    append!(out, hexof.(m_col.value))
    return out
end

if !isfile(MATRIX_ORACLE_BIN)
    @info "oracle SparseMatrix absent — tests ignorés (nécessite oracle/build.sh)"
else
@testset "oracle C++ — SparseMatrix (source gelée)" begin
    rng = MersenneTwister(20260918)
    cases = [random_matrix_case(rng) for _ ∈ 1:40]
    # Branche `cleanup!` (annulations exactes), hyper-sparse puis dense.
    push!(cases, cleanup_matrix_case(false), cleanup_matrix_case(true))
    input = IOBuffer()
    println(input, length(cases))
    for c ∈ cases
        write_matrix_case(input, c)
    end
    text = run_matrix_oracle(String(take!(input)))
    tokens = split(text)
    pos = 1
    problems = String[]
    for (icase, c) ∈ enumerate(cases)
        expected = port_matrix_case(c)
        for (ie, token) ∈ enumerate(expected)
            got = tokens[pos]
            if got != token
                push!(problems, "cas $icase jeton $ie : $token ≠ $got")
            end
            pos += 1
        end
    end
    pos == length(tokens) + 1 ||
        push!(problems, "flux non consommé : $pos ≠ $(length(tokens) + 1)")
    @test isempty(problems)
    isempty(problems) || @info "écarts" problems[1:min(end, 10)]
end
end
