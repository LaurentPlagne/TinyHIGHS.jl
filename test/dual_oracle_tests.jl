# Oracle « solveur » pour M3b/M3c : le LP est résolu par HiGHS — libhighs de la
# source gelée compilée SANS contraction FMA (`install/highs-oracle`, cf. §5 de
# la fiche) — dans la configuration qui correspond au port : simplexe dual
# Dantzig, presolve et échelles LP désactivés (échelles = M5), sans perturbation
# de coûts. Comparaison statut / itérations / objectif, valeurs primales
# bit-à-bit, duals à 1e-12 relatif après convention de sens.
#
# Le corpus couvre aussi des cas > 50 itérations (ré-inversions par horloge
# synthétique) et des cas infaisables (preuve d'infaisabilité).

const DUAL_ORACLE_BIN = abspath(joinpath(@__DIR__, "..", "oracle", "build",
    "dual_oracle"))

"""LP au format de `dual_oracle.cpp` (tableaux 0-based, flottants en %a)."""
function write_dual_lp(io::IO, lp::SimplexLp)
    m = lp.a_matrix
    println(io, lp.num_col, " ", lp.num_row, " ", length(m.index))
    println(io, join(hexof.(lp.col_cost), " "))
    println(io, join(hexof.(lp.col_lower), " "))
    println(io, join(hexof.(lp.col_upper), " "))
    println(io, join(hexof.(lp.row_lower), " "))
    println(io, join(hexof.(lp.row_upper), " "))
    println(io, join(m.start .- 1, " "))
    println(io, join(m.index .- 1, " "))
    println(io, join(hexof.(m.value), " "))
    println(io, Int(lp.sense), " ", hexof(lp.offset))
    return nothing
end

"""
    run_dual_oracle(lp; scale = 0, edge = 0, perturb = 0.0,
                    weight_error_threshold = -1.0, primal_edge = -1)

Sortie de l'oracle : statut, objectif, itérations et vecteurs solution.
`scale` : `simplex_scale_strategy` ; `edge` : stratégie de poids duale
(0 Dantzig, -1 choose, 1 Devex, 2 steepest edge) ; `perturb` :
`dual_simplex_cost_perturbation_multiplier` ; `weight_error_threshold` :
seuil de bascule DSE → Devex (`< 0` : défaut HiGHS) ; `primal_edge` :
stratégie de poids primale (`< 0` : défaut HiGHS), utilisée par la
classification `kUnboundedOrInfeasible` et le nettoyage dual.
"""
function run_dual_oracle(lp::SimplexLp; scale::Int=0, edge::Int=0,
    perturb::Float64=0.0, weight_error_threshold::Float64=-1.0,
    primal_edge::Int=-1)
    input = IOBuffer()
    write_dual_lp(input, lp)
    out = IOBuffer()
    process = run(pipeline(
        `$DUAL_ORACLE_BIN $scale $edge $perturb 0 $weight_error_threshold $primal_edge`;
        stdin=IOBuffer(String(take!(input))), stdout=out))
    success(process) || error("oracle dual : code $(process.exitcode)")
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
    port_solve(lp)

Séquence M3b : moteur, `initialise_for_solve!`, puis `solve!` s'il reste à
faire. Rend statut, objectif, itérations, compteurs de phase et valeurs en
convention d'interface (duals `sense * workDual` pour les colonnes,
`-sense * workDual` pour les logiques converties en lignes).
"""
function port_solve(lp::SimplexLp;
    options::SimplexOptions=SimplexOptions(
        ; dual_simplex_cost_perturbation_multiplier=0.0))
    e = SimplexEngine(lp, options)
    initialise_for_solve!(e)
    if e.model_status != TinyHiGHS.kOptimal
        d = DualSolver(e)
        solve!(d)
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
        phase1=e.info.dual_phase1_iteration_count,
        phase2=e.info.dual_phase2_iteration_count,
        colvalue=xcol, rowvalue=-xrow,
        coldual=sense .* e.info.workDual[1:n],
        rowdual=(-sense) .* e.info.workDual[n+1:n+m])
end

"""
Écarts entre le port et l'oracle pour un cas : statut, itérations, objectif, et
— si optimal — valeurs primales **bit-à-bit** et duals à 1e-12 relatif (dans la
convention de sens d'`Highs_getSolution`, déjà appliquée par `port_solve`).
"""
function dual_case_problems(icase::Int, lp::SimplexLp, ours, theirs;
    check_iterations::Bool=true, check_duals::Bool=true)
    problems = String[]
    ours.status == theirs.status ||
        push!(problems, "cas $icase statut : $(ours.status) ≠ $(theirs.status)")
    if check_iterations
        ours.iterations == theirs.iterations ||
            push!(problems, "cas $icase itérations : $(ours.iterations) ≠ $(theirs.iterations)")
    end
    abs(ours.objective - theirs.objective) <=
        1e-12 + 1e-12 * abs(theirs.objective) ||
        push!(problems, "cas $icase objectif : $(ours.objective) ≠ $(theirs.objective)")
    if ours.status == Int(TinyHiGHS.kOptimal)
        for key ∈ ("colvalue", "rowvalue")
            getproperty(ours, Symbol(key)) == theirs.values[key] ||
                push!(problems, "cas $icase $key : valeur primale non identique")
        end
        if check_duals
            for key ∈ ("coldual", "rowdual")
                err = maximum(abs.(getproperty(ours, Symbol(key)) .-
                                    theirs.values[key]) ./
                              (1 .+ abs.(theirs.values[key])); init=0.0)
                err <= 1e-12 || push!(problems, "cas $icase $key : écart $err")
            end
        end
    end
    return problems
end

"""Cas borné : colonnes finies, ligne faisable par construction."""
function make_bounded_case(rng::AbstractRNG; num_row_max::Int=8)
    num_row = rand(rng, 1:num_row_max)
    num_col = num_row + rand(rng, 0:4)
    A = zeros(num_row, num_col)
    for j ∈ 1:num_row
        A[j, j] = (rand(rng) < 0.5 ? 1.0 : -1.0) * (2 + rand(rng))
        for _ ∈ 1:rand(rng, 0:2)
            A[rand(rng, 1:num_row), j] += randn(rng)
        end
    end
    for j ∈ (num_row + 1):num_col, _ ∈ 1:rand(rng, 0:2)
        A[rand(rng, 1:num_row), j] += randn(rng)
    end
    x0 = [rand(rng) * 10 for _ ∈ 1:num_col]
    col_lower = zeros(num_col)
    col_upper = zeros(num_col)
    for j ∈ 1:num_col
        if rand(rng) < 0.15                       # fixe
            col_lower[j] = x0[j]
            col_upper[j] = x0[j]
        else
            col_lower[j] = x0[j] - rand(rng)
            col_upper[j] = x0[j] + rand(rng)
        end
    end
    ax = A * x0
    row_lower = [ax[i] - rand(rng) for i ∈ 1:num_row]
    row_upper = [ax[i] + rand(rng) for i ∈ 1:num_row]
    for i ∈ 1:num_row
        r = rand(rng)
        r < 0.15 && (row_lower[i] = -kHighsInf)
        r ≥ 0.15 && r < 0.3 && (row_upper[i] = kHighsInf)
    end
    col_cost = [rand(rng) < 0.25 ? 0.0 :
                (rand(rng) < 0.3 ? 3.0 : randn(rng) * 5) for _ ∈ 1:num_col]
    sense = rand(rng) < 0.3 ? kMaximize : kMinimize
    return SimplexLp(num_col, num_row, csc_from_dense(A), col_cost, col_lower,
        col_upper, row_lower, row_upper; offset=randn(rng), sense=sense)
end

"""
Cas à colonne libre : l'infaisabilité duale d'une variable libre ne se
corrige ni par flip ni par shift, donc la phase 1 duale fait des itérations.
Le LP reste borné : la colonne libre a un coefficient non nul dans des lignes
à bornes finies.
"""
function make_free_column_case(rng::AbstractRNG)
    num_row = rand(rng, 1:3)
    num_col = num_row + rand(rng, 1:2)
    A = zeros(num_row, num_col)
    for j ∈ 1:num_col
        A[mod1(j, num_row), j] = (rand(rng) < 0.5 ? 1.0 : -1.0) * (2 + rand(rng))
        for _ ∈ 1:rand(rng, 0:2)
            A[rand(rng, 1:num_row), j] += randn(rng)
        end
    end
    x0 = [rand(rng) * 5 for _ ∈ 1:num_col]
    col_lower = [x0[j] - rand(rng) for j ∈ 1:num_col]
    col_upper = [x0[j] + rand(rng) for j ∈ 1:num_col]
    col_lower[1] = -kHighsInf                    # libre
    col_upper[1] = kHighsInf
    ax = A * x0
    row_lower = [ax[i] - rand(rng) for i ∈ 1:num_row]
    row_upper = [ax[i] + rand(rng) for i ∈ 1:num_row]
    col_cost = [rand(rng) < 0.4 ? 0.0 : randn(rng) * 5 for _ ∈ 1:num_col]
    return SimplexLp(num_col, num_row, csc_from_dense(A), col_cost, col_lower,
        col_upper, row_lower, row_upper; sense=kMinimize)
end

"""
Cas long : transport équilibré à colonnes unilatérales (`x ≥ 0`, lignes
d'égalité). Pas de flip possible, et le simplexe dual enchaîne ~2,5(m+n)
pivots : à `m = 25` le seuil de ré-inversion synthétique (50 updates) est
franchi.
"""
function make_transport_case(rng::AbstractRNG; m::Int=25)
    num_row = 2 * m
    num_col = m * m
    a_start = Int[1]
    a_index = Int[]
    a_value = Float64[]
    col_cost = Float64[]
    for i ∈ 1:m, j ∈ 1:m
        push!(a_index, i)
        push!(a_value, 1.0)
        push!(a_index, m + j)
        push!(a_value, 1.0)
        push!(a_start, length(a_index) + 1)
        push!(col_cost, rand(rng) * 10)
    end
    supply = rand(rng, 1:5, m)
    demand = rand(rng, 1:5, m)
    demand = max.(round.(Int, demand .* (sum(supply) / sum(demand))), 1)
    row = Float64.(vcat(supply, demand))
    mtx = SparseMatrix(num_col, num_row, a_start, a_index, a_value)
    return SimplexLp(num_col, num_row, mtx, col_cost, zeros(num_col),
        fill(kHighsInf, num_col), row, copy(row))
end

"""Cas infaisable : une ligne hors de l'atteignable des bornes de colonnes."""
function make_infeasible_case(rng::AbstractRNG)
    num_row = rand(rng, 1:6)
    num_col = num_row + rand(rng, 0:3)
    A = zeros(num_row, num_col)
    for j ∈ 1:num_row
        A[j, j] = (rand(rng) < 0.5 ? 1.0 : -1.0) * (2 + rand(rng))
        for _ ∈ 1:rand(rng, 0:2)
            A[rand(rng, 1:num_row), j] += randn(rng)
        end
    end
    x0 = [rand(rng) * 10 for _ ∈ 1:num_col]
    col_lower = [x0[j] - rand(rng) for j ∈ 1:num_col]
    col_upper = [x0[j] + rand(rng) for j ∈ 1:num_col]
    ax = A * x0
    reach = sum(abs.(A[1, :]) .* (col_upper .- col_lower))
    row_lower = [ax[i] - rand(rng) for i ∈ 1:num_row]
    row_upper = [ax[i] + rand(rng) for i ∈ 1:num_row]
    row_lower[1] = ax[1] + reach + 1.0
    row_upper[1] = row_lower[1] + 1.0
    return SimplexLp(num_col, num_row, csc_from_dense(A),
        randn(rng, num_col) * 5, col_lower, col_upper, row_lower, row_upper)
end

@testset "oracle HiGHS — dual M3b (Dantzig, sans échelle ni perturbation)" begin
    @test isfile(DUAL_ORACLE_BIN) ||
          error("oracle absent : lancer julia_simplex/oracle/build.sh")
    if isfile(DUAL_ORACLE_BIN)
        rng = MersenneTwister(20260927)
        cases = vcat([make_bounded_case(rng) for _ ∈ 1:20],
            [make_free_column_case(rng) for _ ∈ 1:5],
            [make_bounded_case(rng; num_row_max=18) for _ ∈ 1:5],
            [make_transport_case(rng; m=15)],
            [make_transport_case(rng; m=25)],
            [make_infeasible_case(rng) for _ ∈ 1:10])
        problems = String[]
        phase1_seen = 0
        phase2_seen = 0
        maximize_seen = 0
        fixed_seen = 0
        infinite_seen = 0
        max_iterations = 0
        dantzig_options = SimplexOptions(
            ; dual_simplex_cost_perturbation_multiplier=0.0,
            simplex_dual_edge_weight_strategy=0)
        for (icase, lp) ∈ enumerate(cases)
            ours = port_solve(lp; options=dantzig_options)
            theirs = run_dual_oracle(lp)
            append!(problems, dual_case_problems(icase, lp, ours, theirs))
            ours.phase1 > 0 && (phase1_seen += 1)
            ours.phase2 > 0 && (phase2_seen += 1)
            lp.sense == kMaximize && (maximize_seen += 1)
            any(lp.col_lower .== lp.col_upper) && (fixed_seen += 1)
            any(isinf, lp.row_lower) || any(isinf, lp.row_upper) ||
                (infinite_seen += 1)
            max_iterations = max(max_iterations, ours.iterations)
        end
        @test isempty(problems)
        isempty(problems) || @info "écarts dual" problems[1:min(end, 10)]
        # Couverture : les deux phases, les deux sens, colonnes fixes, bornes de
        # ligne infinies, et au moins un cas > 50 itérations (ré-inversions).
        @test phase1_seen > 0 && phase2_seen > 0
        @test maximize_seen > 0 && fixed_seen > 0 && infinite_seen > 0
        @test max_iterations > 50
    end
end

@testset "oracle HiGHS — dual avec perturbation de coûts (défauts HiGHS)" begin
    if isfile(DUAL_ORACLE_BIN)
        rng = MersenneTwister(20260933)
        cases = vcat([make_bounded_case(rng) for _ ∈ 1:20],
            [make_free_column_case(rng) for _ ∈ 1:5],
            [make_bounded_case(rng; num_row_max=18) for _ ∈ 1:3],
            [make_transport_case(rng; m=15)],
            [make_infeasible_case(rng) for _ ∈ 1:10])
        problems = String[]
        for (icase, lp) ∈ enumerate(cases)
            # Défauts HiGHS : multiplicateur de perturbation 1.0, Dantzig (la
            # comparaison DSE a son propre test), échelles LP désactivées (M5).
            ours = port_solve(lp; options=SimplexOptions(
                ; simplex_dual_edge_weight_strategy=0))
            theirs = run_dual_oracle(lp; perturb=1.0)
            append!(problems, dual_case_problems(icase, lp, ours, theirs))
        end
        @test isempty(problems)
        isempty(problems) || @info "écarts dual perturbé" problems[1:min(end, 10)]
    end
end

@testset "oracle HiGHS — dual DSE (poids steepest edge)" begin
    if isfile(DUAL_ORACLE_BIN)
        rng = MersenneTwister(20260936)
        # Cas généraux : statut, itérations, primal bit-à-bit et duals à 1e-12.
        cases = vcat([make_bounded_case(rng) for _ ∈ 1:20],
            [make_free_column_case(rng) for _ ∈ 1:5],
            [make_infeasible_case(rng) for _ ∈ 1:5])
        dse_options = SimplexOptions(
            ; dual_simplex_cost_perturbation_multiplier=0.0,
            simplex_dual_edge_weight_strategy=2)
        problems = String[]
        for (icase, lp) ∈ enumerate(cases)
            ours = port_solve(lp; options=dse_options)
            theirs = run_dual_oracle(lp; edge=2)
            append!(problems, dual_case_problems(icase, lp, ours, theirs))
        end
        # Cas dégénérés (transport) : mêmes statut/objectif/primal, mais le
        # nombre d'itérations peut différer d'un cran car un écart dual de
        # 1 ULP présent dès le Dantzig y bascule une égalité (cf. fiche).
        for (icase, lp) ∈ enumerate([make_transport_case(rng; m=8),
            make_transport_case(rng; m=12)])
            ours = port_solve(lp; options=dse_options)
            theirs = run_dual_oracle(lp; edge=2)
            append!(problems, dual_case_problems(icase, lp, ours, theirs;
                check_iterations=false, check_duals=false))
        end
        @test isempty(problems)
        isempty(problems) || @info "écarts dual DSE" problems[1:min(end, 10)]
    end
end

@testset "oracle HiGHS — Devex et bascule DSE → Devex" begin
    if isfile(DUAL_ORACLE_BIN)
        rng = MersenneTwister(20260940)
        cases = vcat([make_bounded_case(rng) for _ ∈ 1:15],
            [make_free_column_case(rng) for _ ∈ 1:5],
            [make_infeasible_case(rng) for _ ∈ 1:5])
        problems = String[]
        # Stratégie Devex d'emblée.
        devex_options = SimplexOptions(
            ; dual_simplex_cost_perturbation_multiplier=0.0,
            simplex_dual_edge_weight_strategy=1)
        for (icase, lp) ∈ enumerate(cases)
            ours = port_solve(lp; options=devex_options)
            theirs = run_dual_oracle(lp; edge=1)
            append!(problems, dual_case_problems(icase, lp, ours, theirs))
        end
        # Bascule DSE → Devex forcée par un seuil d'erreur de poids nul (le
        # premier écart non nul la déclenche) ; comportement identique attendu.
        rng_switch = MersenneTwister(20260941)
        switch_options = SimplexOptions(
            ; dual_simplex_cost_perturbation_multiplier=0.0,
            simplex_dual_edge_weight_strategy=-1,
            dual_steepest_edge_weight_log_error_threshold=0.0)
        switched = 0
        for (icase, lp) ∈ enumerate([make_bounded_case(rng_switch) for _ ∈ 1:10])
            ours = port_solve(lp; options=switch_options)
            theirs = run_dual_oracle(lp; edge=-1, weight_error_threshold=0.0)
            append!(problems, dual_case_problems(icase, lp, ours, theirs))
            # Rejoue le cas pour observer le mode final du solveur.
            e = SimplexEngine(lp, switch_options)
            initialise_for_solve!(e)
            if e.model_status != TinyHiGHS.kOptimal
                d = DualSolver(e)
                solve!(d)
                d.edge_weight_mode == TinyHiGHS.kEdgeWeightDevex &&
                    (switched += 1)
            end
        end
        @test isempty(problems)
        isempty(problems) || @info "écarts Devex" problems[1:min(end, 10)]
        @test switched > 0          # la bascule a bien été exercée
    end
end

@testset "oracle HiGHS — défauts complets (choose + perturbation)" begin
    if isfile(DUAL_ORACLE_BIN)
        # Seule l'échelle LP (M5) reste désactivée : stratégie « choose » (DSE
        # avec bascule Devex autorisée, qui ne se déclenche pas sur le corpus)
        # et multiplicateur de perturbation 1.0 des deux côtés.
        rng = MersenneTwister(20260938)
        cases = vcat([make_bounded_case(rng) for _ ∈ 1:15],
            [make_free_column_case(rng) for _ ∈ 1:5])
        problems = String[]
        for (icase, lp) ∈ enumerate(cases)
            ours = port_solve(lp; options=SimplexOptions())
            theirs = run_dual_oracle(lp; edge=-1, perturb=1.0)
            append!(problems, dual_case_problems(icase, lp, ours, theirs))
        end
        @test isempty(problems)
        isempty(problems) || @info "écarts dual défauts" problems[1:min(end, 10)]
    end
end

@testset "le comparateur voit une mutation (coût relatif 1e-6)" begin
    rng = MersenneTwister(20260932)
    lp = make_bounded_case(rng)
    # Dantzig des deux côtés (l'oracle par défaut), échelles et perturbation
    # désactivées.
    dantzig_options = SimplexOptions(
        ; dual_simplex_cost_perturbation_multiplier=0.0,
        simplex_dual_edge_weight_strategy=0)
    ours0 = port_solve(lp; options=dantzig_options)
    theirs0 = run_dual_oracle(lp)
    @test ours0.status == theirs0.status && ours0.iterations == theirs0.iterations
    # Mutation au-delà de la tolérance du comparateur (1e-12 relative) : un coût
    # non nul multiplié par (1 + 1e-6).
    cost = copy(lp.col_cost)
    j = findfirst(!iszero, cost)
    cost[j] *= 1.0 + 1e-6
    lp_mutated = SimplexLp(lp.num_col, lp.num_row, lp.a_matrix, cost,
        lp.col_lower, lp.col_upper, lp.row_lower, lp.row_upper;
        offset=lp.offset, sense=lp.sense)
    theirs1 = run_dual_oracle(lp_mutated)
    @test theirs1.objective != theirs0.objective ||
          theirs1.iterations != theirs0.iterations ||
          theirs1.status != theirs0.status
    # Le port suit la mutation ; l'écart port/oracle reste sous la tolérance.
    ours1 = port_solve(lp_mutated; options=dantzig_options)
    @test ours1.status == theirs1.status && ours1.iterations == theirs1.iterations
    @test abs(ours1.objective - theirs1.objective) <=
          1e-12 + 1e-12 * abs(theirs1.objective)
end

@testset "DualSolver — reset! bit-à-bit et non-divergence" begin
    rng = MersenneTwister(20260940)
    for _ in 1:5
        lp = make_bounded_case(rng)
        e1 = SimplexEngine(lp)
        initialise_for_solve!(e1)
        d_fresh = DualSolver(e1)
        solve!(d_fresh)

        e2 = SimplexEngine(lp)
        initialise_for_solve!(e2)
        d_reused = DualSolver(e2)
        solve!(d_reused)

        # Réinitialisation et deuxième résolution sur le même solveur réutilisé
        initialise_for_solve!(e2)
        reset!(d_reused, e2)
        solve!(d_reused)

        @test e1.iteration_count == e2.iteration_count
        @test e1.model_status == e2.model_status
        @test e1.info.primal_objective_value === e2.info.primal_objective_value
        @test e1.basis.basicIndex == e2.basis.basicIndex
        @test all(e1.info.workValue .=== e2.info.workValue)
        @test all(e1.info.workDual .=== e2.info.workDual)
    end
end
