# Oracle `HFactor` : comparaison bit-à-bit du port avec la source C++ INCHANGÉE.
# Phases couvertes : build, FTRAN/BTRAN, chaînes d'updates Forrest-Tomlin,
# refactorisation (`rebuild`). Voir
# docs/architecture/portage-julia-simplexe-highs.md §4.1.

const HFACTOR_ORACLE_BIN = abspath(joinpath(@__DIR__, "..", "oracle", "build",
    "hfactor_oracle"))

"""Cas d'oracle : matrice, base, RHS, updates `(iRow, aq, ep)` et rebuild."""
struct FactorCase
    num_col::Int
    num_row::Int
    a_start::Vector{Int}
    a_index::Vector{Int}
    a_value::Vector{Float64}
    basic_index::Vector{Int}
    rhs_list::Vector{Vector{Float64}}
    updates::Vector{Tuple{Int,HVector,HVector}}
    do_rebuild::Bool
end

function write_packed(io::IO, v::HVector)
    println(io, v.size, " ", v.count)
    println(io, join((v.index[i] - 1 for i ∈ 1:max(v.count, 0)), " "))
    println(io, join(hexof.(v.array), " "))
    println(io, v.packCount)
    println(io, join((v.packIndex[i] - 1 for i ∈ 1:v.packCount), " "))
    println(io, join((hexof(v.packValue[i]) for i ∈ 1:v.packCount), " "))
    return nothing
end

function write_hfactor_case(io::IO, c::FactorCase)
    println(io, c.num_row, " ", c.num_col, " ", length(c.basic_index), " ",
        length(c.a_index))
    println(io, join(c.a_start .- 1, " "))
    println(io, join(c.a_index .- 1, " "))
    println(io, join(hexof.(c.a_value), " "))
    println(io, join(c.basic_index .- 1, " "))
    println(io, length(c.rhs_list))
    for rhs ∈ c.rhs_list
        println(io, join(hexof.(rhs), " "))
    end
    println(io, length(c.updates))
    for (iRow, aq, ep) ∈ c.updates
        println(io, iRow - 1)
        write_packed(io, aq)
        write_packed(io, ep)
    end
    println(io, c.do_rebuild ? 1 : 0)
    return nothing
end

function run_hfactor_oracle(input::String)
    out = IOBuffer()
    err = IOBuffer()
    process = run(pipeline(`$HFACTOR_ORACLE_BIN`; stdin=IOBuffer(input),
        stdout=out, stderr=err))
    success(process) ||
        error("oracle HFactor : code $(process.exitcode) ; $(String(take!(err)))")
    return String(take!(out))
end

function read_int_vector(tokens, pos::Int)
    n = parse(Int, tokens[pos])
    values = [parse(Int, tokens[pos + i]) for i ∈ 1:n]
    return values, pos + n + 1
end

function read_double_vector(tokens, pos::Int)
    n = parse(Int, tokens[pos])
    values = [tokens[pos + i] for i ∈ 1:n]
    return values, pos + n + 1
end

"""État oracle d'une phase : `(rank, ints, values, pos)`."""
function read_factor_state(tokens, pos::Int)
    rank = parse(Int, tokens[pos])
    pos += 1
    # Ordre exact du dump C++ (entiers et flottants interleavés).
    ints = Vector{Vector{Int}}(undef, 8)
    values = Vector{Vector{String}}(undef, 3)
    ints[1], pos = read_int_vector(tokens, pos)      # l_start
    ints[2], pos = read_int_vector(tokens, pos)      # l_index
    values[1], pos = read_double_vector(tokens, pos) # l_value
    ints[3], pos = read_int_vector(tokens, pos)      # l_pivot_index
    ints[4], pos = read_int_vector(tokens, pos)      # u_pivot_index
    values[2], pos = read_double_vector(tokens, pos) # u_pivot_value
    ints[5], pos = read_int_vector(tokens, pos)      # u_start
    ints[6], pos = read_int_vector(tokens, pos)      # u_last_p
    ints[7], pos = read_int_vector(tokens, pos)      # u_index
    values[3], pos = read_double_vector(tokens, pos) # u_value
    ints[8], pos = read_int_vector(tokens, pos)      # basic_index
    return rank, ints, values, pos
end

function read_solves(tokens, pos::Int, num_row::Int, n_rhs::Int)
    lines = Vector{Vector{String}}()
    for _ ∈ 1:(2 * n_rhs)
        push!(lines, [tokens[pos + i - 1] for i ∈ 1:num_row])
        pos += num_row
    end
    return lines, pos
end

# Champs comparés, dans l'ordre du dump C++ ; indices en 0-based, valeurs en %a.
function factor_state_tokens(f::HFactor)
    return (ints=(copy(f.l_start) .- 1, copy(f.l_index) .- 1,
            copy(f.l_pivot_index) .- 1, copy(f.u_pivot_index) .- 1,
            copy(f.u_start) .- 1, copy(f.u_last_p) .- 1, copy(f.u_index) .- 1,
            copy(f.basic_index) .- 1),
        values=(hexof.(f.l_value), hexof.(f.u_pivot_value), hexof.(f.u_value)))
end

function julia_solve_lines(f::HFactor, rhs_list, num_row::Int)
    lines = Vector{Vector{String}}()
    for rhs ∈ rhs_list
        v = HVector(num_row)
        v.count = -1
        v.array .= rhs
        ftranCall!(f, v, 1.0)
        push!(lines, hexof.(v.array))
        w = HVector(num_row)
        w.count = -1
        w.array .= rhs
        btranCall!(f, w, 1.0)
        push!(lines, hexof.(w.array))
    end
    return lines
end

function compare_phase(f::HFactor, ints, values, solves, rhs_list, num_row, tag)
    problems = String[]
    ours = factor_state_tokens(f)
    names = ("l_start", "l_index", "l_pivot_index", "u_pivot_index", "u_start",
        "u_last_p", "u_index", "basic_index")
    for (i, name) ∈ enumerate(names)
        ours.ints[i] == ints[i] ||
            push!(problems, "$tag $name : $(ours.ints[i]) ≠ $(ints[i])")
    end
    for (i, name) ∈ enumerate(("l_value", "u_pivot_value", "u_value"))
        ours.values[i] == values[i] ||
            push!(problems, "$tag $name : $(ours.values[i]) ≠ $(values[i])")
    end
    julia = julia_solve_lines(f, rhs_list, num_row)
    julia == solves || push!(problems, "$tag solves : $(julia) ≠ $solves")
    return problems
end

"""Audit d'un cas : consomme le dump oracle à partir de `pos` et compare."""
function audit_hfactor_case(ic::Int, tokens, pos::Int, c::FactorCase)
    problems = String[]
    f = HFactor(c.num_col, c.num_row, c.num_row, c.a_start, c.a_index, c.a_value,
        c.basic_index)
    rank_julia = build!(f)

    rank, ints, values, pos = read_factor_state(tokens, pos)
    solves, pos = read_solves(tokens, pos, c.num_row, length(c.rhs_list))
    rank_julia == rank || push!(problems, "cas $ic rank : $rank_julia ≠ $rank")
    if rank_julia == rank
        append!(problems,
            compare_phase(f, ints, values, solves, c.rhs_list, c.num_row, "cas $ic"))
    end
    if rank == 0 && rank_julia == 0
        for (iRow, aq, ep) ∈ c.updates
            update!(f, aq, ep, iRow)
        end
        if !isempty(c.updates)
            _, ints, values, pos = read_factor_state(tokens, pos)
            solves, pos = read_solves(tokens, pos, c.num_row, length(c.rhs_list))
            append!(problems,
                compare_phase(f, ints, values, solves, c.rhs_list, c.num_row,
                    "cas $ic updates"))
        end
        if c.do_rebuild
            f.refactor_info.use = true
            rebuild_rank = build!(f)
            _, ints, values, pos = read_factor_state(tokens, pos)
            solves, pos = read_solves(tokens, pos, c.num_row, length(c.rhs_list))
            rebuild_rank == 0 ||
                push!(problems, "cas $ic rebuild rank : $rebuild_rank ≠ 0")
            append!(problems,
                compare_phase(f, ints, values, solves, c.rhs_list, c.num_row,
                    "cas $ic rebuild"))
        end
    else
        # Base déficiente (ou rang divergent) : les phases supplémentaires ne
        # sont pas comparées, mais leurs jetons doivent être consommés pour
        # garder l'alignement. Les solves initiaux l'ont déjà été plus haut.
        if !isempty(c.updates)
            _, _, _, pos = read_factor_state(tokens, pos)
            _, pos = read_solves(tokens, pos, c.num_row, length(c.rhs_list))
        end
        if c.do_rebuild && isempty(c.updates)
            _, _, _, pos = read_factor_state(tokens, pos)
            _, pos = read_solves(tokens, pos, c.num_row, length(c.rhs_list))
        end
    end
    return problems, pos
end

# --- génération des cas -------------------------------------------------------

function build_case(rng::AbstractRNG, num_col::Int, num_row::Int,
    a_start::Vector{Int}, a_index::Vector{Int}, a_value::Vector{Float64},
    basic_index::Vector{Int}; n_rhs::Int=2, n_updates::Int=0,
    do_rebuild::Bool=false)
    rhs_list = [randn(rng, num_row) for _ ∈ 1:n_rhs]
    updates = Tuple{Int,HVector,HVector}[]
    rows_available = collect(1:num_row)
    for _ ∈ 1:n_updates
        iRow = rand(rng, rows_available)
        aq = HVector(num_row)
        aq.array[iRow] = 1.0 + rand(rng)
        aq.index[1] = iRow
        aq.count = 1
        for _ ∈ 1:rand(rng, 0:2)
            r = rand(rng, rows_available)
            r == iRow && continue
            aq.count += 1
            aq.index[aq.count] = r
            aq.array[r] = randn(rng)
        end
        tight!(aq)
        aq.packFlag = true
        pack!(aq)
        ep = HVector(num_row)
        ep.count = rand(rng, 1:min(3, num_row))
        for k ∈ 1:ep.count
            r = rand(rng, rows_available)
            ep.index[k] = r
            ep.array[r] = randn(rng)
        end
        tight!(ep)
        ep.packFlag = true
        pack!(ep)
        push!(updates, (iRow, aq, ep))
    end
    return FactorCase(num_col, num_row, a_start, a_index, a_value, basic_index,
        rhs_list, updates, do_rebuild)
end

function random_basis_case(rng::AbstractRNG, n::Int)
    n_logical = rand(rng, 0:min(3, max(n - 1, 0)))
    n_struct_basic = n - n_logical
    num_col = rand(rng, n_struct_basic:min(n_struct_basic + 2, n))
    rows = shuffle(rng, 1:n)
    row_of = Dict{Int,Int}()
    for (j, r) ∈ enumerate(rows[1:n_struct_basic])
        row_of[j] = r
    end
    a_start = Int[1]
    a_index = Int[]
    a_value = Float64[]
    unit_col = n_struct_basic >= 1 && rand(rng) < 0.5 ?
               rand(rng, 1:n_struct_basic) : 0
    for j ∈ 1:num_col
        entries = Tuple{Int,Float64}[]
        if j <= n_struct_basic
            if j == unit_col
                push!(entries, (row_of[j], 1.0))
            else
                push!(entries, (row_of[j], (rand(rng) < 0.5 ? 1.0 : -1.0) *
                                          (2.0 + rand(rng))))
                for _ ∈ 1:rand(rng, 0:2)
                    r = rand(rng, 1:n)
                    any(e -> e[1] == r, entries) && continue
                    push!(entries, (r, randn(rng)))
                end
            end
        else
            for _ ∈ 1:rand(rng, 1:2)
                r = rand(rng, 1:n)
                any(e -> e[1] == r, entries) && continue
                push!(entries, (r, randn(rng)))
            end
        end
        sort!(entries; by=first)
        for (r, v) ∈ entries
            push!(a_index, r)
            push!(a_value, v)
        end
        push!(a_start, length(a_index) + 1)
    end
    struct_basic = collect(1:n_struct_basic)
    logicals = collect(num_col + 1:num_col + n_logical)
    basic_index = shuffle(rng, vcat(struct_basic, logicals))
    # Une base déficiente ne reçoit ni update ni rebuild : le C++ aurait alors
    # `refactor_info` invalide, et la source ne définit pas ce chemin.
    probe = HFactor(num_col, n, n, a_start, a_index, a_value, basic_index)
    deficient = build!(probe) != 0
    mode = deficient ? 0 : rand(rng, 1:3)
    n_updates = mode == 1 ? rand(rng, 1:3) : 0
    do_rebuild = mode == 3
    return build_case(rng, num_col, n, a_start, a_index, a_value, basic_index;
        n_updates=n_updates, do_rebuild=do_rebuild)
end

if !isfile(HFACTOR_ORACLE_BIN)
    @info "oracle HFactor absent — tests ignorés (nécessite oracle/build.sh)"
else
@testset "oracle C++ — HFactor (source gelée)" begin
    if isfile(HFACTOR_ORACLE_BIN)
        @testset "étalonnage : build + solves + updates + rebuild" begin
            rng = MersenneTwister(20260916)
            cases = FactorCase[]
            # Cas triviaux : identité (pivots unitaires) et kernel 2×2, avec
            # updates pour l'un et rebuild pour l'autre.
            push!(cases, build_case(rng, 3, 3, [1, 2, 3, 4], [1, 2, 3],
                [1.0, 1.0, 1.0], [1, 2, 3]; n_updates=2))
            push!(cases, build_case(rng, 2, 2, [1, 3, 5], [1, 2, 1, 2],
                [2.0, 1.0, 1.0, 3.0], [1, 2]; do_rebuild=true))
            for _ ∈ 1:40
                n = rand(rng, 2:12)
                push!(cases, random_basis_case(rng, n))
            end

            input = IOBuffer()
            println(input, length(cases))
            for c ∈ cases
                write_hfactor_case(input, c)
            end
            text = run_hfactor_oracle(String(take!(input)))

            tokens = split(text)
            problems = String[]
            pos = 1
            for (ic, c) ∈ enumerate(cases)
                try
                    case_problems, pos = audit_hfactor_case(ic, tokens, pos, c)
                    append!(problems, case_problems)
                catch err
                    push!(problems,
                        "cas $ic : exception $err (pos=$pos, ntok=$(length(tokens)))")
                    break
                end
            end
            pos == length(tokens) + 1 ||
                push!(problems, "flux oracle non consommé : $pos ≠ $(length(tokens) + 1)")
            @test isempty(problems)
            isempty(problems) || @info "écarts" problems[1:min(end, 20)]
        end

        @testset "carence de rang : base complétée par des logiques" begin
            # Colonne nulle : rang déficient ; la source complète par des
            # logiques (buildHandleRankDeficiency/buildMarkSingC).
            c = FactorCase(2, 2, [1, 2, 3], [1, 1], [1.0, 0.0], [1, 2],
                [[1.0, 2.0]], Tuple{Int,HVector,HVector}[], false)
            input = IOBuffer()
            println(input, 1)
            write_hfactor_case(input, c)
            tokens = split(run_hfactor_oracle(String(take!(input))))
            problems, pos = audit_hfactor_case(1, tokens, 1, c)
            @test pos == length(tokens) + 1
            @test isempty(problems)
            isempty(problems) || @info "écarts" problems
        end
    end
end
end
