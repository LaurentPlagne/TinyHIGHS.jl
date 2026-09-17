# Oracle « solveur » pour M4a/M4b : le LP est résolu par HiGHS — libhighs de la
# source gelée compilée SANS contraction FMA (`install/highs-oracle`, cf. §5 de
# la fiche) — dans la configuration qui correspond au port : simplexe primal
# Dantzig (`simplex_strategy = primal`,
# `simplex_primal_edge_weight_strategy = dantzig`), presolve et échelles LP
# désactivés (échelles = M5).
# Comparaison statut / itérations / objectif, et **les quatre vecteurs
# solution bit-à-bit** : primal aux statuts optimal, infaisable et non borné,
# duals aux statuts optimal et infaisable (`HEkk::returnFromSolve` recale les
# duals sur les coûts du LP dans ce dernier cas).
#
# Le corpus couvre : la phase 2 nue (base logique primalement faisable),
# les flips de bornes (BFRT), les violations sous tolérance (shifts de bornes
# puis nettoyage dual de la fin de phase 2), les colonnes libres, la phase 1
# (infaisables), la non-bornitude, et la classification
# `kUnboundedOrInfeasible` → primal (le dual seul ne tranche pas).

const PRIMAL_ORACLE_BIN = DUAL_ORACLE_BIN

"""
    run_primal_oracle(lp)

Sortie de l'oracle en configuration primale : statut, objectif, itérations et
vecteurs solution. `edge`/`cost_perturbation` portent sur le dual (inutilisés
tant que la stratégie est primale) ; `primal_edge` et
`primal_bound_perturbation` sur le primal.
"""
function run_primal_oracle(lp::SimplexLp; scale::Int=0, edge::Int=0,
    cost_perturbation::Float64=0.0, primal_edge::Int=0,
    primal_bound_perturbation::Float64=0.0, strategy::Int=4)
    input = IOBuffer()
    write_dual_lp(input, lp)
    out = IOBuffer()
    process = run(pipeline(
        `$PRIMAL_ORACLE_BIN $scale $edge $cost_perturbation 0 -1 $primal_edge $primal_bound_perturbation $strategy`;
        stdin=IOBuffer(String(take!(input))), stdout=out))
    success(process) || error("oracle primal : code $(process.exitcode)")
    lines = split(strip(String(take!(out))), '\n')
    head = split(lines[1])
    values = Dict{String,Vector{Float64}}()
    for l ∈ lines[2:end]
        tokens = split(l)
        values[tokens[1]] = [parse(Float64, x) for x ∈ tokens[2:end]]
    end
    return (status=parse(Int, head[1]), objective=parse(Float64, head[2]),
        iterations=parse(Int, head[3]), values=values)
end

"""
    port_primal_solve(lp; perturb = 0.0, edge = 0)

Séquence M4a/M4b/M4c : moteur, `initialise_for_solve!`, puis `solve!` primal
s'il reste à faire, avec la stratégie de poids `edge` (0 Dantzig, 1 Devex,
2 steepest edge, -1 choose → Devex) et le multiplicateur de perturbation des
bornes `perturb`. Rend statut, objectif, itérations, compteurs de phase,
drapeaux de couverture (flips, shifts, nettoyage dual) et valeurs en convention
d'interface (`getSolution`).
"""
function port_primal_solve(lp::SimplexLp; perturb::Float64=0.0, edge::Int=0)
    e = SimplexEngine(lp, SimplexOptions(
        ; simplex_primal_edge_weight_strategy=edge,
        primal_simplex_bound_perturbation_multiplier=perturb))
    initialise_for_solve!(e)
    if e.model_status != TinyHiGHS.kOptimal
        p = PrimalSolver(e)
        solve!(p)
    end
    compute_primal_objective_value!(e)
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
    return (status=Int(e.model_status), objective=e.info.primal_objective_value,
        iterations=e.iteration_count,
        phase1=e.info.primal_phase1_iteration_count,
        phase2=e.info.primal_phase2_iteration_count,
        flips=e.info.primal_bound_swap,
        shifted=e.info.bounds_shifted || !e.info.allow_bound_perturbation,
        nested_dual_iter=e.info.dual_phase2_iteration_count,
        colvalue=xcol, rowvalue=-xrow,
        coldual=sense .* e.info.workDual[1:n],
        rowdual=(-sense) .* e.info.workDual[n+1:n+m])
end

"""
Écarts port/oracle d'un cas primal : statut, itérations, objectif, et les
quatre vecteurs solution. Le primal est comparé **bit-à-bit** aux statuts
optimal, infaisable et non borné ; les duals bit-à-bit si infaisable
(`returnFromSolve` les recale sur les coûts du LP) et à 1e-12 relatif si
optimal. Lève une erreur explicite si le port refuse un cas que l'oracle ne
déclare ni infaisable ni non borné.
"""
function primal_case_problems(icase::Int, ours, theirs;
    check_iterations::Bool=true)
    problems = String[]
    optimal = Int(TinyHiGHS.kOptimal)
    infeasible = Int(TinyHiGHS.kInfeasible)
    unbounded = Int(TinyHiGHS.kUnbounded)
    ours.status == theirs.status ||
        push!(problems, "cas $icase statut : $(ours.status) ≠ $(theirs.status)")
    if check_iterations
        ours.iterations == theirs.iterations ||
            push!(problems, "cas $icase itérations : $(ours.iterations) ≠ $(theirs.iterations)")
    end
    abs(ours.objective - theirs.objective) <=
        1e-12 + 1e-12 * abs(theirs.objective) ||
        push!(problems, "cas $icase objectif : $(ours.objective) ≠ $(theirs.objective)")
    if ours.status ∈ (optimal, infeasible, unbounded)
        for key ∈ ("colvalue", "rowvalue")
            getproperty(ours, Symbol(key)) == theirs.values[key] ||
                push!(problems, "cas $icase $key : valeur primale non identique")
        end
    end
    if ours.status == optimal
        for key ∈ ("coldual", "rowdual")
            err = maximum(abs.(getproperty(ours, Symbol(key)) .-
                                theirs.values[key]) ./
                          (1 .+ abs.(theirs.values[key])); init=0.0)
            err <= 1e-12 || push!(problems, "cas $icase $key : écart $err")
        end
    elseif ours.status == infeasible
        for key ∈ ("coldual", "rowdual")
            getproperty(ours, Symbol(key)) == theirs.values[key] ||
                push!(problems, "cas $icase $key : dual non identique")
        end
    end
    return problems
end

"""
Cas où la base logique est primalement faisable : bornes de lignes encadrant 0,
colonnes minorées en 0, coûts des deux signes. Le primal démarre donc en
phase 2 ; les bornes supérieures finies gardent le LP borné.
"""
function make_primal_feasible_case(rng::AbstractRNG; num_row_max::Int=8)
    num_row = rand(rng, 1:num_row_max)
    num_col = num_row + rand(rng, 0:4)
    A = zeros(num_row, num_col)
    for j ∈ 1:num_col, _ ∈ 1:rand(rng, 1:3)
        A[rand(rng, 1:num_row), j] += randn(rng)
    end
    for j ∈ 1:num_col
        sum(abs, @view A[:, j]) == 0.0 && (A[rand(rng, 1:num_row), j] = 1.0)
    end
    col_lower = zeros(num_col)
    col_upper = [rand(rng) * (rand(rng) < 0.3 ? 0.5 : 3.0) + 1e-3
                 for _ ∈ 1:num_col]
    row_lower = [-rand(rng) * 2 for _ ∈ 1:num_row]
    row_upper = [rand(rng) * 2 for _ ∈ 1:num_row]
    for i ∈ 1:num_row
        r = rand(rng)
        r < 0.2 && (row_lower[i] = -kHighsInf)
        (r >= 0.2 && r < 0.35) && (row_upper[i] = kHighsInf)
    end
    col_cost = [rand(rng) < 0.2 ? 0.0 :
                (rand(rng) < 0.4 ? -rand(rng) * 4 : randn(rng))
                for _ ∈ 1:num_col]
    sense = rand(rng) < 0.3 ? kMaximize : kMinimize
    return SimplexLp(num_col, num_row, csc_from_dense(A), col_cost, col_lower,
        col_upper, row_lower, row_upper; offset=randn(rng), sense=sense)
end

"""
Cas avec une violation de borne sous la tolérance sur une ou deux lignes : le
primal est forcé en phase 2, les valeurs de base violées sont absorbées par des
shifts de bornes, puis `cleanup` les retire — ce qui peut déclencher le
nettoyage dual de fin de phase 2 (chemin croisé M4a).

La violation n'est appliquée que si l'autre borne l'accepte : `Highs::run`
rejette avant tout simplexe un LP dont une borne inférieure dépasse sa borne
supérieure (`infeasibleBoundsOk`), contrôle de niveau LP que le port ne porte
pas (cf. fiche, §5).
"""
function make_primal_small_violation_case(rng::AbstractRNG)
    lp = make_primal_feasible_case(rng; num_row_max=6)
    num_row, num_col = lp.num_row, lp.num_col
    row_lower = copy(lp.row_lower)
    row_upper = copy(lp.row_upper)
    # Au moins une ligne doit porter une violation : la première ligne finie.
    finie = findfirst(i -> row_lower[i] > -1e10 && row_upper[i] < 1e10, 1:num_row)
    finie === nothing && return lp
    for i ∈ (finie, rand(rng, 1:num_row))
        (row_lower[i] > -1e10 && row_upper[i] < 1e10) || continue
        δ = 1.5e-7 + rand(rng) * 2e-4
        if rand(rng) < 0.5
            row_upper[i] >= δ && (row_lower[i] = δ)
        else
            row_lower[i] <= -δ && (row_upper[i] = -δ)
        end
    end
    return SimplexLp(num_col, num_row, lp.a_matrix, lp.col_cost, lp.col_lower,
        lp.col_upper, row_lower, row_upper; offset=lp.offset, sense=lp.sense)
end

"""
Cas à colonnes libres : la colonne libre ne peut ni flipper ni se poser sur
une borne ; elle entre et sort par `removeNonbasicFreeColumn`. Le LP reste
borné si chaque colonne libre a un coefficient non nul dans une ligne finie.
"""
function make_primal_free_case(rng::AbstractRNG)
    num_row = rand(rng, 2:6)
    num_col = num_row + rand(rng, 1:3)
    A = zeros(num_row, num_col)
    for j ∈ 1:num_col, _ ∈ 1:rand(rng, 1:3)
        A[rand(rng, 1:num_row), j] += randn(rng)
    end
    for j ∈ 1:num_col
        sum(abs, @view A[:, j]) == 0.0 && (A[rand(rng, 1:num_row), j] = 1.0)
    end
    num_free = rand(rng, 1:min(2, num_col))
    col_lower = zeros(num_col)
    col_upper = [rand(rng) * 3 + 1e-3 for _ ∈ 1:num_col]
    for j ∈ randperm(rng, num_col)[1:num_free]
        col_lower[j] = -kHighsInf
        col_upper[j] = kHighsInf
    end
    row_lower = [-rand(rng) * 2 for _ ∈ 1:num_row]
    row_upper = [rand(rng) * 2 for _ ∈ 1:num_row]
    col_cost = [rand(rng) < 0.3 ? 0.0 :
                (rand(rng) < 0.5 ? -1.0 : 1.0) * rand(rng) * 4
                for _ ∈ 1:num_col]
    sense = rand(rng) < 0.3 ? kMaximize : kMinimize
    return SimplexLp(num_col, num_row, csc_from_dense(A), col_cost, col_lower,
        col_upper, row_lower, row_upper; offset=randn(rng), sense=sense)
end

"""
Cas non borné : une colonne sans borne supérieure et de coût −1 n'apparaît que
dans des lignes sans borne supérieure, donc rien ne bloque sa croissance ; la
base logique est primalement faisable, le primal certifie en phase 2.
"""
function make_primal_unbounded_case(rng::AbstractRNG)
    num_row = rand(rng, 1:5)
    jf = rand(rng, 1:3)
    num_col = jf + rand(rng, 0:2)
    A = zeros(num_row, num_col)
    for j ∈ 1:num_col, _ ∈ 1:rand(rng, 1:2)
        A[rand(rng, 1:num_row), j] += randn(rng)
    end
    for j ∈ 1:num_col
        sum(abs, @view A[:, j]) == 0.0 && (A[rand(rng, 1:num_row), j] = 1.0)
    end
    col_lower = zeros(num_col)
    col_upper = [rand(rng) * 3 + 1e-3 for _ ∈ 1:num_col]
    col_upper[jf] = kHighsInf
    col_cost = randn(rng, num_col) * 0.1
    col_cost[jf] = -1.0
    row_lower = [-rand(rng) * 2 for _ ∈ 1:num_row]
    row_upper = [rand(rng) < 0.6 ? kHighsInf : rand(rng) * 2
                 for _ ∈ 1:num_row]
    any(row_upper .== kHighsInf) ||
        (row_upper[rand(rng, 1:num_row)] = kHighsInf)
    for i ∈ 1:num_row
        A[i, jf] = row_upper[i] == kHighsInf ? 1.0 : 0.0
    end
    return SimplexLp(num_col, num_row, csc_from_dense(A), col_cost, col_lower,
        col_upper, row_lower, row_upper; sense=kMinimize)
end

"""
Cas faisable ou infaisable mais non borné, à base logique primalement
infaisable : le dual seul conclut `kUnboundedOrInfeasible` (il ne peut prouver
l'infaisabilité primale), et le primal de classification doit d'abord résorber
l'infaisabilité avant de trancher.
"""
function make_unbounded_or_infeasible_case(rng::AbstractRNG)
    num_row = rand(rng, 2:6)
    jf = rand(rng, 1:3)
    num_col = max(2, jf + rand(rng, 0:2))
    A = zeros(num_row, num_col)
    for j ∈ 1:num_col, _ ∈ 1:rand(rng, 1:2)
        A[rand(rng, 1:num_row), j] += randn(rng)
    end
    for j ∈ 1:num_col
        sum(abs, @view A[:, j]) == 0.0 && (A[rand(rng, 1:num_row), j] = 1.0)
    end
    col_lower = zeros(num_col)
    col_upper = [rand(rng) * 3 + 1e-3 for _ ∈ 1:num_col]
    col_upper[jf] = kHighsInf
    col_cost = randn(rng, num_col) * 0.1
    col_cost[jf] = -1.0
    row_lower = [-rand(rng) * 2 for _ ∈ 1:num_row]
    row_upper = [rand(rng) < 0.5 ? kHighsInf : rand(rng) * 2
                 for _ ∈ 1:num_row]
    any(row_upper .== kHighsInf) ||
        (row_upper[rand(rng, 1:num_row)] = kHighsInf)
    # La colonne non bornée ne peut être bloquée que par une borne inférieure.
    for i ∈ 1:num_row
        A[i, jf] = row_upper[i] == kHighsInf ? abs(A[i, jf]) + 0.5 : 0.0
    end
    # Base logique infaisable : une ligne à borne inférieure positive, sans
    # la colonne non bornée.
    j2 = jf == 1 ? 2 : 1
    i0 = rand(rng, 1:num_row)
    A[i0, :] .= 0.0
    A[i0, j2] = 1.0
    row_lower[i0] = 0.5 + rand(rng)
    row_upper[i0] = kHighsInf
    j2 > jf && (col_upper[j2] = kHighsInf)
    return SimplexLp(num_col, num_row, csc_from_dense(A), col_cost, col_lower,
        col_upper, row_lower, row_upper; sense=kMinimize)
end

"""
    port_dual_solve(lp; options)

Séquence M3b + classification M4b : moteur, `initialise_for_solve!`, puis le
dual s'il reste à faire, y compris le primal de classification. Rend statut,
itérations totales et itérations primales de classification (couverture).
"""
function port_dual_solve(lp::SimplexLp; options::SimplexOptions=SimplexOptions())
    e = SimplexEngine(lp, options)
    initialise_for_solve!(e)
    if e.model_status != TinyHiGHS.kOptimal
        solve!(DualSolver(e))
    end
    return (status=Int(e.model_status), iterations=e.iteration_count,
        primal_iterations=e.info.primal_phase1_iteration_count +
                          e.info.primal_phase2_iteration_count)
end

"""
    compare_primal_corpus(gen, ncases, seed)

Compare un corpus au primal oracle et rend `(problems, refus, couverture)` :
un refus (erreur explicite du port) est compté dans `refus` si le statut de
l'oracle figure dans `expected_refusals`, sinon dans `problems`. Depuis M4b/M4c
le port traite infaisables et non bornés ; `expected_refusals` ne sert plus que
de filet pour les cas que M4c ne couvrirait pas.
"""
function compare_primal_corpus(gen, ncases::Int, seed::Int;
    check_iterations::Bool=true, expected_refusals::Tuple=())
    rng = MersenneTwister(seed)
    problems = String[]
    refus = Int[]
    couverture = (flips=0, shifted=0, nested_dual=0)
    for icase ∈ 1:ncases
        lp = gen(rng)
        theirs = run_primal_oracle(lp)
        ours = try
            port_primal_solve(lp)
        catch err
            message = sprint(showerror, err)
            theirs.status ∈ expected_refusals ||
                push!(problems,
                    "cas $icase refus inattendu (statut oracle $(theirs.status)) : $message")
            push!(refus, theirs.status)
            continue
        end
        append!(problems, primal_case_problems(icase, ours, theirs;
            check_iterations=check_iterations))
        couverture = (flips=couverture.flips + (ours.flips > 0),
            shifted=couverture.shifted + ours.shifted,
            nested_dual=couverture.nested_dual +
                        (ours.nested_dual_iter > 0))
    end
    return problems, refus, couverture
end

if !isfile(PRIMAL_ORACLE_BIN)
    @info "oracle primal absent — tests ignorés (nécessite oracle/build.sh)"
else
@testset "primal Dantzig — base logique primalement faisable" begin
        problems, refus, couverture = compare_primal_corpus(
            make_primal_feasible_case, 40, 12345)
        @test isempty(problems)
        isempty(problems) || @info "écarts primal faisable" problems[1:min(end, 10)]
        isempty(refus) || @info "cas refusés" refus
        # Le corpus atteint bien les flips BFRT (branche réécrite).
        @test couverture.flips > 0
    end

    @testset "primal Dantzig — violations sous tolérance (shifts, nettoyage dual)" begin
        problems, refus, couverture = compare_primal_corpus(
            make_primal_small_violation_case, 60, 31337;
            expected_refusals=(8,))
        @test isempty(problems)
        isempty(problems) || @info "écarts primal shifts" problems[1:min(end, 10)]
        # Les violations sous tolérance exercent les shifts de bornes, et le
        # nettoyage dual de fin de phase 2 tourne sur une partie du corpus.
        @test couverture.shifted > 0
        @test couverture.nested_dual > 0
    end

    @testset "primal Dantzig — colonnes libres" begin
        problems, refus, couverture = compare_primal_corpus(
            make_primal_free_case, 30, 777; expected_refusals=(8, 10))
        @test isempty(problems)
        isempty(problems) || @info "écarts primal colonnes libres" problems[1:min(end, 10)]
    end

    @testset "primal Dantzig — infaisables (phase 1)" begin
        rng = MersenneTwister(20261001)
        for (label, perturb) ∈ (("sans perturbation", 0.0),
            ("bornes perturbées", 1.0))
            cases = [make_infeasible_case(rng) for _ ∈ 1:40]
            problems = String[]
            phase1_seen = 0
            for (icase, lp) ∈ enumerate(cases)
                ours = port_primal_solve(lp; perturb=perturb)
                theirs = run_primal_oracle(lp;
                    primal_bound_perturbation=perturb)
                append!(problems, primal_case_problems(icase, ours, theirs))
                ours.status == Int(TinyHiGHS.kInfeasible) ||
                    push!(problems,
                        "cas $icase : statut $(ours.status), infaisable attendu")
                ours.phase1 > 0 && (phase1_seen += 1)
            end
            @test isempty(problems)
            isempty(problems) ||
                @info "écarts primal infaisable ($label)" problems[1:min(end, 10)]
            # La phase 1 a réellement itéré (elle n'a pas conclu au rebuild).
            @test phase1_seen > 0
        end
    end

    @testset "primal Dantzig — non bornés" begin
        rng = MersenneTwister(20261002)
        for (label, perturb) ∈ (("sans perturbation", 0.0),
            ("bornes perturbées", 1.0))
            cases = [make_primal_unbounded_case(rng) for _ ∈ 1:20]
            problems = String[]
            for (icase, lp) ∈ enumerate(cases)
                ours = port_primal_solve(lp; perturb=perturb)
                theirs = run_primal_oracle(lp;
                    primal_bound_perturbation=perturb)
                append!(problems, primal_case_problems(icase, ours, theirs))
            end
            @test isempty(problems)
            isempty(problems) ||
                @info "écarts primal non borné ($label)" problems[1:min(end, 10)]
        end
    end

    @testset "primal Devex / steepest edge / choose" begin
        rng = MersenneTwister(20261010)
        # Les transports (colonnes unilatérales, égalités) font plusieurs
        # dizaines d'itérations : les poids y changent réellement le chemin
        # (65 / 52 / 60 itérations Dantzig / Devex / steepest à m = 15).
        cases = vcat([make_primal_feasible_case(rng) for _ ∈ 1:12],
            [make_primal_feasible_case(rng; num_row_max=18) for _ ∈ 1:5],
            [make_primal_small_violation_case(rng) for _ ∈ 1:12],
            [make_primal_free_case(rng) for _ ∈ 1:8],
            [make_primal_unbounded_case(rng) for _ ∈ 1:8],
            [make_transport_case(rng; m=12), make_transport_case(rng; m=15)])
        for (label, edge) ∈ (("Devex", 1), ("steepest edge", 2),
            ("choose → Devex", -1))
            problems = String[]
            for (icase, lp) ∈ enumerate(cases)
                ours = port_primal_solve(lp; edge=edge)
                theirs = run_primal_oracle(lp; primal_edge=edge)
                append!(problems, primal_case_problems(icase, ours, theirs))
            end
            @test isempty(problems)
            isempty(problems) ||
                @info "écarts primal ($label)" problems[1:min(end, 10)]
        end
    end

    @testset "classification kUnboundedOrInfeasible → primal" begin
        # Le dual seul conclut `kUnboundedOrInfeasible` ; `HEkk::solve` lance
        # alors le primal pour trancher (M4b, critère resté ouvert au M3c).
        rng = MersenneTwister(20261003)
        cases = [make_unbounded_or_infeasible_case(rng) for _ ∈ 1:20]
        # Dantzig d'abord, puis la stratégie primale par défaut (`choose` →
        # Devex, M4c) : c'est la configuration réelle de `HEkk::solve`.
        for (label, primal_edge, oracle_edge) ∈
            (("Dantzig", 0, 0), ("choose → Devex", -1, -1))
            options = SimplexOptions(
                ; dual_simplex_cost_perturbation_multiplier=0.0,
                simplex_dual_edge_weight_strategy=0,
                simplex_primal_edge_weight_strategy=primal_edge)
            # Le premier cas, dual seul : statut non résolu avant classification.
            e_dual = SimplexEngine(cases[1], options)
            initialise_for_solve!(e_dual)
            if e_dual.model_status != TinyHiGHS.kOptimal
                solve!(DualSolver(e_dual); classify=false)
            end
            @test e_dual.model_status == TinyHiGHS.kUnboundedOrInfeasible
            problems = String[]
            classified = 0
            for (icase, lp) ∈ enumerate(cases)
                ours = port_dual_solve(lp; options=options)
                theirs = run_dual_oracle(lp; edge=0,
                    primal_edge=oracle_edge)
                ours.status == theirs.status ||
                    push!(problems,
                        "cas $icase statut : $(ours.status) ≠ $(theirs.status)")
                ours.iterations == theirs.iterations ||
                    push!(problems,
                        "cas $icase itérations : $(ours.iterations) ≠ $(theirs.iterations)")
                ours.primal_iterations > 0 && (classified += 1)
            end
            @test isempty(problems)
            isempty(problems) ||
                @info "écarts classification ($label)" problems[1:min(end, 10)]
            # Sur une partie du corpus, la classification fait itérer le primal.
            @test classified > 0
        end
    end
end
