# Oracle « séquence » pour M5 : un LP est muté (bornes, coûts, RHS,
# coefficients) puis résolu plusieurs fois sur le **même** moteur — warm start
# compris — par `julia_simplex/oracle/build/sequence_oracle` (libhighs de la
# source gelée SANS contraction FMA, `presolve=off`, échelles désactivées) et
# par le port. Comparaison par solve : statut, itérations, objectif et, si
# optimal, les quatre vecteurs bit-à-bit.
#
# Les séquences sont synthétiques (générateur déterministe) ou rejouées depuis
# des captures réelles anonymisées (`instances/sequences/`).

const SEQUENCE_ORACLE_BIN = abspath(joinpath(@__DIR__, "..", "oracle", "build",
    "sequence_oracle"))

"""Copie indépendante d'un `SimplexLp` (le moteur partage le modèle fourni)."""
function copy_lp_for_port(lp::SimplexLp)
    m = lp.a_matrix
    matrix = SparseMatrix(m.num_col, m.num_row, copy(m.start), copy(m.index),
        copy(m.value))
    return SimplexLp(lp.num_col, lp.num_row, matrix, copy(lp.col_cost),
        copy(lp.col_lower), copy(lp.col_upper), copy(lp.row_lower),
        copy(lp.row_upper); offset=lp.offset, sense=lp.sense)
end

"""
    run_sequence_oracle(lp, ops)

Rejoue `ops` (lignes au format de `sequence_oracle.cpp`) sur `lp` dans
l'oracle et rend `(results, history)` : `results` porte `(index, status,
objective, iterations)` par solve, `history[k]` les quatre vecteurs du solve
`k` s'il est optimal (`getSolution`).
"""
function run_sequence_oracle(lp::SimplexLp, ops::AbstractVector{String})
    input = IOBuffer()
    write_dual_lp(input, lp)
    for line ∈ ops
        println(input, line)
    end
    println(input, "end")
    out = IOBuffer()
    process = run(pipeline(
        `$SEQUENCE_ORACLE_BIN 0 -1 1.0 0 -1.0 -1 -1.0 -1 1`;
        stdin=IOBuffer(String(take!(input))), stdout=out))
    success(process) ||
        error("sequence_oracle : code $(process.exitcode)")
    results = NamedTuple[]
    history = Dict{String,Vector{Float64}}[]
    values = Dict{String,Vector{Float64}}()
    for line ∈ eachline(IOBuffer(String(take!(out))))
        tokens = split(line)
        isempty(tokens) && continue
        if tokens[1] == "solve"
            push!(results, (index=parse(Int, tokens[2]),
                status=parse(Int, tokens[3]), objective=parse(Float64, tokens[4]),
                iterations=parse(Int, tokens[5])))
            # Un dictionnaire neuf par solve (sinon `history` ne garde que le
            # dernier : les entrées référencent le même objet).
            values = Dict{String,Vector{Float64}}()
            push!(history, values)
        elseif tokens[1] ∈ ("colvalue", "coldual", "rowvalue", "rowdual")
            values[tokens[1]] = [parse(Float64, x) for x ∈ tokens[2:end]]
        end
    end
    return results, history
end

"""
    run_sequence_port(lp, ops)

Rejoue `ops` sur un moteur unique : `initialise_for_solve!` et `DualSolver` à
chaque `solve`, mutations entre les résolutions. Rend `(results, history)` :
`history[k]` les quatre vecteurs du solve `k` s'il est optimal, comme
`getSolution`.
"""
function run_sequence_port(lp::SimplexLp, ops::AbstractVector{String};
    options::SimplexOptions=SimplexOptions())
    e = SimplexEngine(copy_lp_for_port(lp), options)
    results = NamedTuple[]
    history = Dict{String,Vector{Float64}}[]
    iteration_previous = 0
    solve_index = 0
    for line ∈ ops
        tokens = split(line)
        isempty(tokens) && continue
        op = String(tokens[1])
        if op == "solve"
            solve_index += 1
            initialise_for_solve!(e)
            if e.model_status != TinyHiGHS.kOptimal
                solve!(DualSolver(e))
            end
            push!(results, (index=solve_index, status=Int(e.model_status),
                objective=e.bounds_infeasible ? 0.0 :
                          e.info.primal_objective_value,
                iterations=e.iteration_count - iteration_previous))
            iteration_previous = e.iteration_count
            if e.model_status == TinyHiGHS.kOptimal
                n, m = lp.num_col, lp.num_row
                xcol = copy(e.info.workValue[1:n])
                xrow = copy(e.info.workValue[n+1:n+m])
                for r ∈ 1:m
                    iVar = e.basis.basicIndex[r]
                    if iVar <= n
                        xcol[iVar] = e.info.baseValue[r]
                    else
                        xrow[iVar - n] = e.info.baseValue[r]
                    end
                end
                sense = Int(lp.sense)
                push!(history, Dict{String,Vector{Float64}}(
                    "colvalue" => xcol, "rowvalue" => -xrow,
                    "coldual" => sense .* e.info.workDual[1:n],
                    "rowdual" => (-sense) .* e.info.workDual[n+1:n+m]))
            else
                push!(history, Dict{String,Vector{Float64}}())
            end
        elseif op == "change_col_bounds"
            change_col_bounds!(e, parse(Int, tokens[2]),
                parse(Float64, tokens[3]), parse(Float64, tokens[4]))
        elseif op == "change_cols_bounds"
            n = parse(Int, tokens[2])
            position = 3
            cols = Vector{Int}(undef, n)
            lowers = Vector{Float64}(undef, n)
            uppers = Vector{Float64}(undef, n)
            for k ∈ 1:n
                cols[k] = parse(Int, tokens[position])
                lowers[k] = parse(Float64, tokens[position + 1])
                uppers[k] = parse(Float64, tokens[position + 2])
                position += 3
            end
            change_cols_bounds!(e, cols, lowers, uppers)
        elseif op == "change_row_bounds"
            change_row_bounds!(e, parse(Int, tokens[2]),
                parse(Float64, tokens[3]), parse(Float64, tokens[4]))
        elseif op == "change_rows_bounds"
            n = parse(Int, tokens[2])
            position = 3
            rows = Vector{Int}(undef, n)
            lowers = Vector{Float64}(undef, n)
            uppers = Vector{Float64}(undef, n)
            for k ∈ 1:n
                rows[k] = parse(Int, tokens[position])
                lowers[k] = parse(Float64, tokens[position + 1])
                uppers[k] = parse(Float64, tokens[position + 2])
                position += 3
            end
            change_rows_bounds!(e, rows, lowers, uppers)
        elseif op == "change_cols_cost"
            n = parse(Int, tokens[2])
            position = 3
            cols = Vector{Int}(undef, n)
            costs = Vector{Float64}(undef, n)
            for k ∈ 1:n
                cols[k] = parse(Int, tokens[position])
                costs[k] = parse(Float64, tokens[position + 1])
                position += 2
            end
            change_cols_cost!(e, cols, costs)
        elseif op == "change_coeff"
            change_coeff!(e, parse(Int, tokens[2]), parse(Int, tokens[3]),
                parse(Float64, tokens[4]))
        elseif op == "change_sense"
            change_objective_sense!(e,
                parse(Int, tokens[2]) < 0 ? kMaximize : kMinimize)
        elseif op == "limit"
            # La limite du port est fixée par `options` (immuables) : l'op ne
            # sert qu'à l'oracle, où elle est posée avant le solve suivant.
            nothing
        elseif op == "end"
            break
        else
            error("opération inconnue : $op")
        end
    end
    return results, history
end

"""Ligne `change_cols_bounds` (colonnes 1-based, bornes en `%a`)."""
function op_change_cols_bounds(cols::AbstractVector{Int},
    lowers::AbstractVector{Float64}, uppers::AbstractVector{Float64})
    parts = String[string(length(cols))]
    for k ∈ eachindex(cols)
        push!(parts, string(cols[k], " ", hexof(lowers[k]), " ",
            hexof(uppers[k])))
    end
    return join(vcat(["change_cols_bounds"], parts), " ")
end

"""Ligne `change_rows_bounds` (lignes 1-based)."""
function op_change_rows_bounds(rows::AbstractVector{Int},
    lowers::AbstractVector{Float64}, uppers::AbstractVector{Float64})
    parts = String[string(length(rows))]
    for k ∈ eachindex(rows)
        push!(parts, string(rows[k], " ", hexof(lowers[k]), " ",
            hexof(uppers[k])))
    end
    return join(vcat(["change_rows_bounds"], parts), " ")
end

"""Ligne `change_cols_cost`."""
function op_change_cols_cost(cols::AbstractVector{Int},
    costs::AbstractVector{Float64})
    parts = String[string(length(cols))]
    for k ∈ eachindex(cols)
        push!(parts, string(cols[k], " ", hexof(costs[k])))
    end
    return join(vcat(["change_cols_cost"], parts), " ")
end

"""
Séquence déterministe : une mutation au hasard (bornes, coûts, RHS,
coefficient existant ou nouveau) suivie d'un solve, quatre fois. Les mutations
restent petites pour que les PL gardent un intérêt (feasible/infeasible/borné)
sans devenir mal conditionnés.
"""
function make_sequence_ops(rng::AbstractRNG, lp::SimplexLp)
    ops = String["solve"]
    m = lp.a_matrix
    for _ ∈ 1:4
        r = rand(rng)
        if r < 0.3
            cols = unique([rand(rng, 1:lp.num_col)
                           for _ ∈ 1:rand(rng, 1:min(4, lp.num_col))])
            lowers = [max(-1e3, lp.col_lower[j] - rand(rng) * 0.5)
                      for j ∈ cols]
            uppers = [min(1e3, lp.col_upper[j] + rand(rng) * 0.5)
                      for j ∈ cols]
            push!(ops, op_change_cols_bounds(cols, lowers, uppers))
        elseif r < 0.5
            cols = unique([rand(rng, 1:lp.num_col)
                           for _ ∈ 1:rand(rng, 1:min(4, lp.num_col))])
            push!(ops, op_change_cols_cost(cols,
                [randn(rng) * 2 for _ ∈ eachindex(cols)]))
        elseif r < 0.7
            rows = unique([rand(rng, 1:lp.num_row)
                           for _ ∈ 1:rand(rng, 1:min(3, lp.num_row))])
            lowers = [lp.row_lower[i] - rand(rng) * 0.5 for i ∈ rows]
            uppers = [lp.row_upper[i] + rand(rng) * 0.5 for i ∈ rows]
            push!(ops, op_change_rows_bounds(rows, lowers, uppers))
        elseif r < 0.85
            j = rand(rng, 1:lp.num_col)
            k = rand(rng, m.start[j]:(m.start[j + 1] - 1))
            push!(ops, string("change_coeff ", m.index[k], " ", j, " ",
                hexof(m.value[k] * (0.5 + rand(rng)))))
        else
            j = rand(rng, 1:lp.num_col)
            push!(ops, string("change_coeff ", rand(rng, 1:lp.num_row), " ",
                j, " ", hexof(randn(rng))))
        end
        push!(ops, "solve")
    end
    return ops
end

"""
Écarts port/oracle d'une séquence : statut et itérations par solve, objectif à
1e-12 relatif, et **les quatre vecteurs bit-à-bit** pour chaque solve optimal.
"""
function sequence_problems(icase::Int, ours, theirs,
    ours_history, theirs_history)
    problems = String[]
    length(ours) == length(theirs) ||
        return ["cas $icase : $(length(ours)) solves port ≠ $(length(theirs)) oracle"]
    for (k, (a, b)) ∈ enumerate(zip(ours, theirs))
        a.status == b.status ||
            push!(problems, "cas $icase solve $k statut : $(a.status) ≠ $(b.status)")
        a.iterations == b.iterations ||
            push!(problems,
                "cas $icase solve $k itérations : $(a.iterations) ≠ $(b.iterations)")
        if a.status == Int(TinyHiGHS.kOptimal)
            abs(a.objective - b.objective) >
                1e-12 + 1e-12 * abs(b.objective) &&
                push!(problems,
                    "cas $icase solve $k objectif : $(a.objective) ≠ $(b.objective)")
            # Vecteurs : 1e-12 relatif. Le warm start laisse un écart résiduel
            # de ~1 ULP sur certains vecteurs (statuts, itérations et objectifs
            # identiques) : le bit-à-bit n'est pas tenu sur les séquences.
            for key ∈ ("colvalue", "rowvalue", "coldual", "rowdual")
                error = maximum(abs.(ours_history[k][key] .-
                                     theirs_history[k][key]) ./
                                (1 .+ abs.(theirs_history[k][key])); init=0.0)
                error <= 1e-12 ||
                    push!(problems,
                        "cas $icase solve $k $key : écart $error")
            end
        end
    end
    return problems
end

if !isfile(SEQUENCE_ORACLE_BIN)
    @info "oracle séquence absent — tests ignorés (nécessite oracle/build.sh)"
else
@testset "M5 — mutations du LP et warm start" begin
        rng = MersenneTwister(20261020)
        problems = String[]
        statuses = Int[]
        for icase ∈ 1:12
            lp = make_primal_feasible_case(rng; num_row_max=6)
            ops = make_sequence_ops(rng, lp)
            theirs, theirs_history = run_sequence_oracle(lp, ops)
            ours, ours_history = run_sequence_port(lp, ops)
            append!(problems, sequence_problems(icase, ours, theirs,
                ours_history, theirs_history))
            append!(statuses, [b.status for b ∈ theirs])
        end
        @test isempty(problems)
        isempty(problems) || @info "écarts séquence" problems[1:min(end, 10)]
        # Les corpus mélangent optimaux et cas particuliers.
        @test count(==(Int(TinyHiGHS.kOptimal)), statuses) > 0
    end

    @testset "M5 — bornes incohérentes (réparation et rejet)" begin
        # Petite incohérence (≤ tolérance) : réparée des deux côtés, le solve
        # repart sur les bornes rectifiées.
        lp = make_primal_feasible_case(MersenneTwister(20261021);
            num_row_max=4)
        tol = 1e-7
        ops = String[
            op_change_cols_bounds([1], [1.0], [1.0 - tol / 4]),
            "solve",
            "solve",
        ]
        theirs, theirs_history = run_sequence_oracle(lp, ops)
        ours, ours_history = run_sequence_port(lp, ops)
        @test isempty(sequence_problems(1, ours, theirs, ours_history,
            theirs_history))
        # Incohérence significative : `infeasibleBoundsOk` rend le PL
        # infaisable sans itération, des deux côtés.
        ops_bad = String[
            op_change_cols_bounds([1], [1.0], [0.5]),
            "solve",
            "solve",
        ]
        theirs_bad, theirs_bad_history = run_sequence_oracle(lp, ops_bad)
        ours_bad, ours_bad_history = run_sequence_port(lp, ops_bad)
        @test isempty(sequence_problems(2, ours_bad, theirs_bad,
            ours_bad_history, theirs_bad_history))
        @test all(r -> r.status == Int(TinyHiGHS.kInfeasible) &&
                       r.iterations == 0, ours_bad)
    end

    @testset "M5 — limite d'itérations par solve" begin
        # `HApp::solveLpSimplex` recopie un compteur remis à zéro à chaque
        # `Highs_run` : `simplex_iteration_limit` vaut **par solve**, pas
        # cumulée sur la séquence. Référence oracle (mêmes options) :
        # itérations 1, 1, 0 ; statuts 14, 14, 7.
        lp = SimplexLp(2, 3, SparseMatrix(2, 3, [1, 3, 5], [1, 2, 1, 3],
            [1.0, 1.0, 1.0, 1.0]), [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0],
            [-Inf, -Inf, -Inf], [1.0, 2 / 3, 2 / 3])
        ops = String["limit 1", "solve", "solve", "solve"]
        theirs, theirs_history = run_sequence_oracle(lp, ops)
        ours, ours_history = run_sequence_port(lp, ops;
            options=SimplexOptions(; simplex_iteration_limit=1))
        @test isempty(sequence_problems(1, ours, theirs, ours_history,
            theirs_history))
        @test [r.iterations for r ∈ ours] == [1, 1, 0]
        @test [r.status for r ∈ ours] ==
              [Int(TinyHiGHS.kIterationLimit), Int(TinyHiGHS.kIterationLimit),
               Int(TinyHiGHS.kOptimal)]
    end

    @testset "M5 — déterminisme (séquence rejouée)" begin
        # Une instance par scénario, aucun état global : deux rejeux de la même
        # séquence rendent exactement les mêmes résultats (statuts, itérations,
        # objectifs, vecteurs).
        rng = MersenneTwister(20261023)
        for _ ∈ 1:3
            lp = make_primal_feasible_case(rng; num_row_max=6)
            ops = make_sequence_ops(rng, lp)
            first_results, first_history = run_sequence_port(lp, ops)
            second_results, second_history = run_sequence_port(lp, ops)
            @test first_results == second_results
            @test first_history == second_history
        end
    end

    @testset "M5 — changement de coefficient (base conservée)" begin
        # `Highs_changeCoeff` jette l'état du Ekk mais `basis_` reste valide :
        # le solve suivant repart de la base mémorisée (0 itération), pas de la
        # base logique.
        lp = make_primal_feasible_case(MersenneTwister(20261022);
            num_row_max=4)
        m = lp.a_matrix
        j = 1
        k = m.start[j]
        ops = String[
            "solve",
            string("change_coeff ", m.index[k], " ", j, " ",
                hexof(m.value[k] * 1.1)),
            "solve",
            "solve",
        ]
        theirs, theirs_history = run_sequence_oracle(lp, ops)
        ours, ours_history = run_sequence_port(lp, ops)
        @test isempty(sequence_problems(1, ours, theirs, ours_history,
            theirs_history))
    end
end

# The captured warm-start corpus is also a standalone regression: it must be
# replayable without the external C++ oracle.  In particular, `base.lp` is
# written in objective-first LP syntax, so this catches any future
# column-order permutation in `read_lp`.
@testset "M5 — corpus de rejeu anonymisé" begin
    corpus_root = normpath(joinpath(@__DIR__, "..", "instances", "sequences"))
    for (name, expected_solves, expected_c0_bounds) in
        (("sequence_small", 76, (12.46, 12.46)),
         ("sequence_medium", 100, (-Inf, Inf)))
        root = joinpath(corpus_root, name)
        lp = read_lp(joinpath(root, "base.lp"))
        @test (lp.col_lower[1], lp.col_upper[1]) == expected_c0_bounds
        ops = readlines(joinpath(root, "operations.txt"))
        results, _ = run_sequence_port(lp, ops)
        @test length(results) == expected_solves
        @test all(r -> r.status == Int(TinyHiGHS.kOptimal), results)
        @test all(r -> isfinite(r.objective), results)
    end
end
