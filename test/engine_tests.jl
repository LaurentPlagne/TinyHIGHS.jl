# Tests M3a — `SimplexEngine` : base logique, bornes et coûts de l'espace de
# travail, valeurs primales/duales, objectifs. L'oracle de cette tranche est
# algébrique (plan §3) : chaque invariant est recalculé en dense depuis la
# matrice du LP, sans réutiliser les routines du port.
using Random

"""Valeurs de toutes les variables : non basiques, puis basiques écrasées."""
function workspace_values(e)
    x = copy(e.info.workValue)
    for iRow ∈ 1:e.lp.num_row
        x[e.basis.basicIndex[iRow]] = e.info.baseValue[iRow]
    end
    return x
end

"""Séquence de valeurs M3a (bornes, coûts, primal, dual, objectifs)."""
function initialise_values!(e)
    initialise_cost!(e, kPrimal, kSolvePhaseUnknown)
    initialise_bound!(e, kPrimal, kSolvePhaseUnknown)
    initialise_nonbasic_value_and_move!(e)
    compute_primal!(e)
    compute_dual!(e)
    compute_primal_objective_value!(e)
    compute_dual_objective_value!(e)
    return e
end

"""
Invariants M3a recalculés en dense : valeur non basique à sa borne selon
`nonbasicMove`, `[A I] x = 0`, `workDual = workCost - [A I]' pi` avec
`pi = -workDual[logiques]` (et `workDual` nul sur les basiques), objectifs.
Rend les écarts maximaux, divisés par l'échelle du cas.
"""
function engine_invariant_errors(e, A::Matrix{Float64})
    num_col, num_row = e.lp.num_col, e.lp.num_row
    x = workspace_values(e)
    bound_error = 0.0
    for iVar ∈ 1:(num_col + num_row)
        e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue || continue
        lower, upper = e.info.workLower[iVar], e.info.workUpper[iVar]
        move = e.basis.nonbasicMove[iVar]
        if move == kNonbasicMoveUp
            bound_error = max(bound_error, abs(x[iVar] - lower))
        elseif move == kNonbasicMoveDn
            bound_error = max(bound_error, abs(x[iVar] - upper))
        elseif lower == upper                      # fixe
            bound_error = max(bound_error, abs(x[iVar] - lower))
        else                                       # libre
            bound_error = max(bound_error, abs(x[iVar]))
        end
    end
    primal_error = 0.0
    for i ∈ 1:num_row
        acc = x[num_col + i]
        for j ∈ 1:num_col
            acc += A[i, j] * x[j]
        end
        primal_error = max(primal_error, abs(acc))
    end
    basic_dual_error = 0.0
    for iRow ∈ 1:num_row
        basic_dual_error = max(basic_dual_error,
            abs(e.info.workDual[e.basis.basicIndex[iRow]]))
    end
    pi = [-e.info.workDual[num_col + i] for i ∈ 1:num_row]
    dual_error = 0.0
    for j ∈ 1:num_col
        ref = e.info.workCost[j]
        for i ∈ 1:num_row
            ref -= A[i, j] * pi[i]
        end
        dual_error = max(dual_error, abs(e.info.workDual[j] - ref))
    end
    for i ∈ 1:num_row
        dual_error = max(dual_error,
            abs(e.info.workDual[num_col + i] - (e.info.workCost[num_col + i] - pi[i])))
    end
    z_p_ref = e.cost_scale *
              sum(e.lp.col_cost[j] * x[j] for j ∈ 1:num_col; init=0.0) +
              e.lp.offset
    z_d_ref = 0.0
    for iVar ∈ 1:(num_col + num_row)
        e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue || continue
        reduced = e.info.workCost[iVar]
        if iVar <= num_col
            reduced -= sum(A[i, iVar] * pi[i] for i ∈ 1:num_row; init=0.0)
        else
            reduced -= pi[iVar - num_col]
        end
        z_d_ref += x[iVar] * reduced
    end
    z_d_ref = e.cost_scale * z_d_ref + Int(e.lp.sense) * e.lp.offset
    scale = 1.0 + max(maximum(abs, A; init=0.0) * maximum(abs, x; init=0.0),
        maximum(abs, e.info.workDual; init=0.0))
    return (; bound_error,
        primal_error=primal_error / scale,
        basic_dual_error=basic_dual_error / scale,
        dual_error=dual_error / scale,
        primal_objective_error=abs(e.info.primal_objective_value - z_p_ref) / scale,
        dual_objective_error=abs(e.info.dual_objective_value - z_d_ref) / scale)
end

"""Bornes aléatoires : fixe, minorée, majorée, boxée ou libre."""
function random_bound_pair(rng::AbstractRNG)
    a, b = randn(rng) * 10, randn(rng) * 10
    kind = rand(rng, 1:5)
    kind == 1 && return (a, a)                    # fixe
    kind == 2 && return (min(a, b), kHighsInf)    # minorée
    kind == 3 && return (-kHighsInf, max(a, b))   # majorée
    kind == 4 && return (min(a, b), max(a, b))    # boxée
    return (-kHighsInf, kHighsInf)                # libre
end

"""
Cas bien conditionné : les lignes des logiques en base et celles des colonnes
structurelles de base sont disjointes, chaque colonne structurelle de base a
une entrée dominante sur sa ligne. Les entrées parasites laissent la base non
singulière en général ; l'appelant écarte les tirages dont `invert!` rend une
carence de rang.
"""
function random_engine_case(rng::AbstractRNG; num_row_max::Int=6,
    sense::ObjSense=kMinimize, cost_scale_factor::Int=0)
    num_row = rand(rng, 1:num_row_max)
    num_col = num_row + rand(rng, 0:2)
    n_logical = rand(rng, 0:min(2, num_row))
    n_struct_basic = num_row - n_logical
    rows = shuffle(rng, 1:num_row)
    logical_rows = rows[1:n_logical]
    struct_rows = rows[(n_logical + 1):end]
    A = zeros(num_row, num_col)
    for j ∈ 1:num_col
        if j <= n_struct_basic
            A[struct_rows[j], j] = (rand(rng) < 0.5 ? 1.0 : -1.0) * (2 + rand(rng))
        end
        for _ ∈ 1:rand(rng, 0:2)
            A[rand(rng, 1:num_row), j] += randn(rng)
        end
    end
    col_cost = [rand(rng) < 0.3 ? 0.0 : randn(rng) * 10 for _ ∈ 1:num_col]
    col_lower = zeros(num_col)
    col_upper = zeros(num_col)
    for j ∈ 1:num_col
        col_lower[j], col_upper[j] = random_bound_pair(rng)
    end
    row_lower = zeros(num_row)
    row_upper = zeros(num_row)
    for i ∈ 1:num_row
        row_lower[i], row_upper[i] = random_bound_pair(rng)
    end
    basic = shuffle(rng,
        vcat(collect(1:n_struct_basic), num_col .+ logical_rows))
    lp = SimplexLp(num_col, num_row, csc_from_dense(A), col_cost, col_lower,
        col_upper, row_lower, row_upper; offset=randn(rng), sense=sense)
    options = SimplexOptions(; cost_scale_factor=cost_scale_factor)
    return lp, options, A, basic
end

"""Base encodée : `basicIndex` fourni, `nonbasicFlag` cohérent."""
function make_basis(lp::SimplexLp, basic::Vector{Int})
    basis = SimplexBasis(lp.num_col, lp.num_row)
    fill!(basis.nonbasicFlag, kNonbasicFlagTrue)
    for (iRow, iVar) ∈ enumerate(basic)
        basis.basicIndex[iRow] = iVar
        basis.nonbasicFlag[iVar] = kNonbasicFlagFalse
    end
    return basis
end

@testset "M3a — base logique et mouvements" begin
    # Colonnes : fixe, minorée, boxée proche de la borne inférieure, boxée
    # proche de la borne supérieure, boxée équilibrée (égalité → Dn), majorée,
    # libre. Lignes : libre et encadrée.
    A = zeros(2, 7)
    col_lower = [-3.0, 2.0, -1.0, -4.0, -2.0, -kHighsInf, -kHighsInf]
    col_upper = [-3.0, kHighsInf, 5.0, 1.0, 2.0, 6.0, kHighsInf]
    lp = SimplexLp(7, 2, csc_from_dense(A), zeros(7), col_lower, col_upper,
        [-kHighsInf, 2.0], [kHighsInf, 6.0])
    e = SimplexEngine(lp)
    @test set_basis!(e) === e
    @test e.basis.basicIndex === e.nla.basic_index === e.nla.factor.basic_index
    @test e.basis.basicIndex == [8, 9]
    @test e.basis.nonbasicFlag == Int8[1, 1, 1, 1, 1, 1, 1, 0, 0]
    # `setBasis` ne renseigne les mouvements que des colonnes ; les logiques
    # basiques restent à Ze (`setup!` remet les tableaux à zéro, là où la
    # source les laisse non initialisés).
    @test e.basis.nonbasicMove ==
          Int8[kNonbasicMoveZe, kNonbasicMoveUp, kNonbasicMoveUp,
        kNonbasicMoveDn, kNonbasicMoveDn, kNonbasicMoveDn, kNonbasicMoveZe,
        kNonbasicMoveZe, kNonbasicMoveZe]
    @test e.info.num_basic_logicals == 2

    # `set_nonbasic_move!` relit les bornes du LP (mêmes mouvements ici) et
    # remet Ze sur les basiques même si leur mouvement était faux.
    e.basis.nonbasicMove[8] = kNonbasicMoveDn
    @test set_nonbasic_move!(e) === e
    @test e.basis.nonbasicMove[8] == kNonbasicMoveZe
    @test e.basis.nonbasicMove[1] == kNonbasicMoveZe
end

@testset "M3a — bornes et coûts de travail" begin
    # Colonnes libre / minorée / majorée / boxée / fixe ; lignes libre et
    # encadrée, pour couvrir les quatre branches de la phase 1 duale.
    A = zeros(2, 5)
    lp = SimplexLp(5, 2, csc_from_dense(A), [1.5, -2.0, 3.0, 0.0, -0.5],
        [-kHighsInf, 3.0, -kHighsInf, 1.0, 7.0],
        [kHighsInf, kHighsInf, 4.0, 2.0, 7.0],
        [-kHighsInf, 2.0], [kHighsInf, 6.0])
    e = SimplexEngine(lp)

    @test initialise_lp_col_bound!(e) === e
    @test initialise_lp_row_bound!(e) === e
    @test e.info.workLower ==
          [-kHighsInf, 3.0, -kHighsInf, 1.0, 7.0, -kHighsInf, -6.0]
    @test e.info.workUpper == [kHighsInf, kHighsInf, 4.0, 2.0, 7.0, kHighsInf, -2.0]
    @test e.info.workRange ==
          [kHighsInf, kHighsInf, kHighsInf, 1.0, 0.0, kHighsInf, 4.0]
    @test all(iszero, e.info.workLowerShift) && all(iszero, e.info.workUpperShift)

    # Phase 2 (ou primal) : bornes du LP inchangées.
    @test initialise_bound!(e, kPrimal, kSolvePhaseUnknown) === e
    @test e.info.workLower[1] == -kHighsInf && e.info.workUpper[1] == kHighsInf
    @test initialise_bound!(e, kDual, kSolvePhase2) === e
    @test e.info.workLower[4] == 1.0 && e.info.workUpper[4] == 2.0

    # Phase 1 duale : libre → [-1000, 1000], majorée → [-1, 0],
    # minorée → [0, 1], boxée ou fixe → [0, 0].
    @test initialise_bound!(e, kDual, kSolvePhase1) === e
    @test e.info.workLower == [-1000.0, 0.0, -1.0, 0.0, 0.0, -1000.0, 0.0]
    @test e.info.workUpper == [1000.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 0.0]
    @test e.info.workRange == [2000.0, 1.0, 1.0, 0.0, 0.0, 2000.0, 0.0]
    @test !e.info.bounds_shifted && !e.info.bounds_perturbed

    # Coûts : sens, échelle par `2^cost_scale_factor`, logiques à zéro.
    e2 = SimplexEngine(lp, SimplexOptions(; cost_scale_factor=2))
    @test initialise_cost!(e2, kPrimal, kSolvePhaseUnknown) === e2
    @test e2.info.workCost[1:5] == [6.0, -8.0, 12.0, 0.0, -2.0]
    @test all(iszero, e2.info.workCost[6:7]) && all(iszero, e2.info.workShift)
    @test !e2.info.costs_shifted && !e2.info.costs_perturbed

    lp_max = SimplexLp(5, 2, csc_from_dense(A), [1.5, -2.0, 3.0, 0.0, -0.5],
        [-kHighsInf, 3.0, -kHighsInf, 1.0, 7.0],
        [kHighsInf, kHighsInf, 4.0, 2.0, 7.0],
        [-kHighsInf, 2.0], [kHighsInf, 6.0]; sense=kMaximize)
    e3 = SimplexEngine(lp_max)
    initialise_cost!(e3, kDual, kSolvePhase2)
    @test e3.info.workCost[1:5] == [-1.5, 2.0, -3.0, 0.0, 0.5]

    # Perturbation des coûts (M3c) : dans la colonne minorée seule, le coût
    # augmente ; libre et fixe ne sont pas perturbées ; la boxée suit le signe
    # de son coût ; les logiques reçoivent (0.5 - r)·1e-12.
    lp_pert = SimplexLp(4, 1, csc_from_dense(zeros(1, 4)), [1.5, -2.0, 3.0, 0.0],
        [3.0, -kHighsInf, 1.0, 7.0], [kHighsInf, kHighsInf, 2.0, 7.0],
        [0.0], [1.0])
    e4 = SimplexEngine(lp_pert, SimplexOptions(
        ; dual_simplex_cost_perturbation_multiplier=1.0))
    initialise_cost!(e4, kDual, kSolvePhaseUnknown; perturb=false)
    @test !e4.info.costs_perturbed
    initialise_cost!(e4, kDual, kSolvePhaseUnknown; perturb=true)
    @test e4.info.costs_perturbed
    info = e4.info
    @test info.workCost[1] > 1.5                    # minorée : +xpert
    @test info.workCost[2] == -2.0                  # libre : intacte
    @test info.workCost[3] > 3.0                    # boxée positive : +xpert
    @test info.workCost[4] == 0.0                   # fixe : intacte
    @test all(0 .< abs.([info.workCost[1] - 1.5, info.workCost[3] - 3.0]) .< 1e-4)
    @test abs(info.workCost[5] - 0.0) < 1e-11       # logique : ±1e-12 près
    # Multiplicateur nul : aucune perturbation.
    e5 = SimplexEngine(lp_pert, SimplexOptions(
        ; dual_simplex_cost_perturbation_multiplier=0.0))
    initialise_cost!(e5, kDual, kSolvePhaseUnknown; perturb=true)
    @test !e5.info.costs_perturbed
    @test e5.info.workCost[1:4] == [1.5, -2.0, 3.0, 0.0]
    # Bornes primales (M4b) : écart relatif `multiplicateur·5e-7` sur les
    # bornes finies, libre intacte, fixe **non basique** intacte, valeurs non
    # basiques recalées sur `nonbasicMove`, bornes des basiques recopiées.
    e6 = SimplexEngine(lp_pert, SimplexOptions(
        ; primal_simplex_bound_perturbation_multiplier=1.0))
    set_basis!(e6)
    set_nonbasic_move!(e6)
    @test initialise_bound!(e6, kPrimal, kSolvePhaseUnknown; perturb=true) === e6
    @test e6.info.bounds_perturbed
    info6 = e6.info
    @test info6.workUpper[1] == kHighsInf
    @test 3.0 * (1 - 5e-7) <= info6.workLower[1] < 3.0
    @test info6.workLower[2] == -kHighsInf && info6.workUpper[2] == kHighsInf
    @test 1.0 * (1 - 5e-7) <= info6.workLower[3] < 1.0
    @test 2.0 < info6.workUpper[3] <= 2.0 * (1 + 5e-7)
    @test info6.workLower[4] == 7.0 && info6.workUpper[4] == 7.0
    @test info6.workValue[1] == info6.workLower[1]   # minorée : mouvement Up
    @test info6.workValue[3] == info6.workLower[3]   # boxée : Up (|lower| < |upper|)
    @test info6.baseLower[1] == info6.workLower[5]   # logique basique
    @test info6.baseUpper[1] == info6.workUpper[5]
    # Multiplicateur nul : aucun écart.
    e7 = SimplexEngine(lp_pert, SimplexOptions(
        ; primal_simplex_bound_perturbation_multiplier=0.0))
    set_basis!(e7)
    @test initialise_bound!(e7, kPrimal, kSolvePhaseUnknown; perturb=true) === e7
    @test !e7.info.bounds_perturbed
    @test e7.info.workLower[1] == 3.0 && e7.info.workUpper[3] == 2.0
end

@testset "M3a — valeurs non basiques initiales" begin
    A = zeros(1, 4)
    # Boxée avec mouvement invalide (corrigé en Up, valeur basse), boxée Dn
    # (valeur haute), libre (valeur nulle), fixe.
    lp = SimplexLp(4, 1, csc_from_dense(A), zeros(4),
        [-1.0, -2.0, -kHighsInf, 5.0], [4.0, 3.0, kHighsInf, 5.0],
        [-kHighsInf], [kHighsInf])
    e = SimplexEngine(lp)
    set_basis!(e)
    initialise_bound!(e, kPrimal, kSolvePhaseUnknown)
    e.basis.nonbasicMove[1] = Int8(42)          # hors {-1, 0, 1} : corrigé en Up
    e.basis.nonbasicMove[2] = kNonbasicMoveDn
    @test initialise_nonbasic_value_and_move!(e) === e
    @test e.basis.nonbasicMove[1:4] ==
          Int8[kNonbasicMoveUp, kNonbasicMoveDn, kNonbasicMoveZe, kNonbasicMoveZe]
    @test e.info.workValue[1:4] == [-1.0, 3.0, 0.0, 5.0]
    @test e.basis.nonbasicMove[5] == kNonbasicMoveZe
    @test e.info.workValue[5] == 0.0            # basique : laissé à zéro
end

@testset "M3a — cas construit optimal (double sens, offset, échelle)" begin
    # min : x1 = 2 (borne inf), x2 = 0, logique s = -A x = 0.5 ; coûts réduits
    # 1 et 0.5 ; base logique optimale, donc z_d = sense * z_p.
    A = [-0.25 0.75]
    for c1 ∈ (1.0, -1.0)
        c2 = 0.5 * c1
        sense = c1 > 0 ? kMinimize : kMaximize
        for csf ∈ (0, 2)
            lp = SimplexLp(2, 1, csc_from_dense(A), [c1, c2], [2.0, 0.0],
                [10.0, 1.0], [-1.0], [1.0]; offset=3.0, sense=sense)
            e = SimplexEngine(lp, SimplexOptions(; cost_scale_factor=csf))
            set_basis!(e)
            @test invert!(e.nla) == 0
            initialise_cost!(e, kPrimal, kSolvePhaseUnknown)
            initialise_bound!(e, kPrimal, kSolvePhaseUnknown)
            initialise_nonbasic_value_and_move!(e)
            compute_primal!(e)
            compute_dual!(e)
            primal = compute_primal_objective_value!(e)
            dual = compute_dual_objective_value!(e)
            scale = 2.0^csf
            z_p = 2.0 * c1 + 3.0
            @test e.info.workValue[1] == 2.0 && e.info.workValue[2] == 0.0
            @test e.info.baseValue[1] == 0.5
            @test e.info.workDual[1] == Int(sense) * scale * c1
            @test e.info.workDual[2] == Int(sense) * scale * c2
            @test primal == z_p
            @test dual == Int(sense) * (2.0 * scale * c1 + 3.0)
            # phase 1 : pas de décalage d'objectif dual.
            @test compute_dual_objective_value!(e, kSolvePhase1) ==
                  Int(sense) * 2.0 * scale * c1
        end
    end
end

@testset "M3a — invariants sur cas aléatoires" begin
    rng = MersenneTwister(20260916)
    checked = 0
    draws = 0
    worst = (; bound_error=0.0, primal_error=0.0, basic_dual_error=0.0,
        dual_error=0.0, primal_objective_error=0.0, dual_objective_error=0.0)
    n_logical_seen = Set{Int}()
    senses_seen = Set{ObjSense}()
    scales_seen = Set{Int}()
    while checked < 60 && draws < 500
        draws += 1
        sense = rand(rng, (kMinimize, kMaximize))
        csf = rand(rng, (0, 1, 3))
        lp, options, A, basic = random_engine_case(rng; sense=sense,
            cost_scale_factor=csf)
        e = SimplexEngine(lp, options; basis=make_basis(lp, basic))
        set_nonbasic_move!(e)
        invert!(e.nla) == 0 || continue        # base rang-déficiente : écartée
        initialise_values!(e)
        err = engine_invariant_errors(e, A)
        worst = (; bound_error=max(worst.bound_error, err.bound_error),
            primal_error=max(worst.primal_error, err.primal_error),
            basic_dual_error=max(worst.basic_dual_error, err.basic_dual_error),
            dual_error=max(worst.dual_error, err.dual_error),
            primal_objective_error=max(worst.primal_objective_error,
                err.primal_objective_error),
            dual_objective_error=max(worst.dual_objective_error,
                err.dual_objective_error))
        push!(n_logical_seen, count(>(lp.num_col), basic))
        push!(senses_seen, sense)
        push!(scales_seen, csf)
        checked += 1
    end
    tol = 1e-8
    @test checked == 60
    @test worst.bound_error == 0.0
    @test worst.primal_error < tol
    @test worst.basic_dual_error < tol
    @test worst.dual_error < tol
    @test worst.primal_objective_error < tol
    @test worst.dual_objective_error < tol
    # Couverture : les deux sens, les trois échelles, des bases sans logique
    # (0) comme avec (1, 2).
    @test length(senses_seen) == 2 && length(scales_seen) == 3
    @test n_logical_seen == Set([0, 1, 2])
    worst.primal_error < tol || @info "écarts M3a" worst
end

@testset "M3a — densités, gardes et tailles dégénérées" begin
    # Base {colonne 1, logique 2} : B = diag(2, 1). Les non basiques valent 0
    # (logique libre et colonne 2 à sa borne inférieure) donc le FTRAN de
    # `compute_primal` n'est pas appelé et `primal_col_density` reste 0. Un seul
    # coût basique non nul : le BTRAN rend un vecteur creux de densité 1/2, et
    # `dual_col_density` suit la moyenne glissante.
    A = [2.0 0.3; 0.0 3.0]
    lp = SimplexLp(2, 2, csc_from_dense(A), [1.0, 0.7], [0.0, 0.0], [10.0, 10.0],
        [-kHighsInf, -kHighsInf], [kHighsInf, kHighsInf])
    e = SimplexEngine(lp; basis=make_basis(lp, [1, 4]))
    set_nonbasic_move!(e)
    @test invert!(e.nla) == 0
    initialise_cost!(e, kPrimal, kSolvePhaseUnknown)
    initialise_bound!(e, kPrimal, kSolvePhaseUnknown)
    initialise_nonbasic_value_and_move!(e)
    compute_primal!(e)
    @test e.info.primal_col_density == 0.0
    @test e.info.dual_col_density == 1.0
    compute_dual!(e)
    @test e.info.dual_col_density == 0.95 * 1.0 + 0.05 * (1 / 2)
    @test TinyHiGHS.update_operation_result_density(1.0, 0.5) == 0.975
    @test engine_invariant_errors(e, A).primal_error < 1e-8
    # `invert_num_el` : entrées de L, de U et pivots (régression `num_row = 1`).
    @test e.nla.factor.invert_num_el == length(e.nla.factor.l_index) +
                                        length(e.nla.factor.u_index) + 2

    # Coûts tous nuls : `dual_col.count == 0`, workDual = workCost et densité
    # inchangée ; l'objectif dual reste défini.
    lp0 = SimplexLp(2, 2, csc_from_dense(A), [0.0, 0.0], [0.0, 0.0], [10.0, 10.0],
        [-kHighsInf, -kHighsInf], [kHighsInf, kHighsInf]; offset=2.0)
    e0 = SimplexEngine(lp0; basis=make_basis(lp0, [1, 2]))
    set_nonbasic_move!(e0)
    @test invert!(e0.nla) == 0
    initialise_values!(e0)
    @test all(iszero, e0.info.workDual)
    @test e0.info.dual_col_density == 1.0
    @test e0.info.dual_objective_value == 2.0
    @test e0.info.primal_objective_value == 2.0
    @test e0.basis.nonbasicMove[[1, 2]] ==
          Int8[kNonbasicMoveZe, kNonbasicMoveZe]

    # num_row = 0 : aucune logique, toutes les colonnes non basiques.
    lp_nr0 = SimplexLp(2, 0, SparseMatrix(2, 0), [2.0, 3.0], [1.0, 4.0],
        [5.0, 6.0], Float64[], Float64[]; offset=-1.0)
    e_nr0 = SimplexEngine(lp_nr0)
    set_basis!(e_nr0)
    @test invert!(e_nr0.nla) == 0
    initialise_for_solve!(e_nr0)
    @test e_nr0.info.workValue == [1.0, 4.0]
    @test e_nr0.info.primal_objective_value == 2.0 * 1.0 + 3.0 * 4.0 - 1.0
    @test e_nr0.info.dual_objective_value == 2.0 * 1.0 + 3.0 * 4.0 - 1.0

    # num_col = 0 : base entièrement logique.
    lp_nc0 = SimplexLp(0, 2, SparseMatrix(0, 2), Float64[], Float64[],
        Float64[], [-1.0, -2.0], [1.0, 2.0]; offset=7.0)
    e_nc0 = SimplexEngine(lp_nc0)
    set_basis!(e_nc0)
    @test invert!(e_nc0.nla) == 0
    initialise_for_solve!(e_nc0)
    @test e_nc0.basis.basicIndex == [1, 2] && e_nc0.info.num_basic_logicals == 2
    @test all(iszero, e_nc0.info.baseValue)
    @test e_nc0.info.primal_objective_value == 7.0
end

@testset "M3c — hachage de base, tabous et backtracking" begin
    # Hachage : une base logique combine ses logiques ; l'échange d'un couple
    # compatible avec `sparse_inverse_combine`/`sparse_combine` revient au même.
    A = [1.0 2.0; 0.0 1.0]
    lp = SimplexLp(2, 2, csc_from_dense(A), [1.0, 1.0], [0.0, 0.0], [1.0, 1.0],
        [-kHighsInf, -kHighsInf], [kHighsInf, kHighsInf])
    e = SimplexEngine(lp)
    set_basis!(e)
    h_logical = e.basis.hash
    @test h_logical != 0 && h_logical == TinyHiGHS.basis_hash(e.basis)
    # Change la base à la main (sort 3, entre 1 ; indices 0-based) et vérifie
    # que le hachage suit les deux sens.
    e.basis.hash = TinyHiGHS.sparse_inverse_combine(h_logical, 2)
    e.basis.hash = TinyHiGHS.sparse_combine(e.basis.hash, 0)
    expected_basis = SimplexBasis(Int[1, 4], Int8[0, 1, 1, 0], Int8[0, 0, 0, 0],
        UInt64(0))
    @test e.basis.hash == TinyHiGHS.basis_hash(expected_basis)

    # Liste de mauvais changements : ajout, tabou, sauvegarde/restauration de
    # rangée, purge des entrées devenues inutiles.
    idx = add_bad_basis_change!(e, 2, 4, 1, kBadBasisChangeSingular, true)
    @test idx == 1 && length(e.bad_basis_change) == 1
    @test taboo_bad_basis_change(e)
    vals = [1.0, 2.0, 3.0]
    apply_taboo_row_out!(e, vals, 0.0)
    @test vals[2] == 0.0 && e.bad_basis_change[1].save_value == 2.0
    unapply_taboo_row_out!(e, vals)
    @test vals == [1.0, 2.0, 3.0]
    clear_bad_basis_change_taboo_flag!(e)
    @test !taboo_bad_basis_change(e)
    # `update_bad_basis_change!` retire un enregistrement quand l'entrée pivot
    # a un effet primal **au moins égal** à la tolérance (prédicat `>= tol` de
    # la source) ; un effet négligeable laisse l'enregistrement en place.
    col_aq = HVector(2)
    col_aq.count = 1
    col_aq.index[1] = 2
    col_aq.array[2] = 1e-12                       # < tolérance : conservé
    update_bad_basis_change!(e, col_aq, 1.0)
    @test length(e.bad_basis_change) == 1
    col_aq.array[2] = 1.0                         # ≥ tolérance : retiré
    update_bad_basis_change!(e, col_aq, 1.0)
    @test isempty(e.bad_basis_change)
    add_bad_basis_change!(e, 2, 4, 1, kBadBasisChangeSingular, true)
    clear_bad_basis_change!(e, kBadBasisChangeSingular)
    @test isempty(e.bad_basis_change)

    # Détection de cyclage : si le hachage candidat a déjà été visité sur
    # l'itération précédente, le changement est marqué mauvais et tabou.
    e2 = SimplexEngine(lp)
    set_basis!(e2)
    e2.iteration_count = 5
    e2.previous_iteration_cycling_detected = 4
    variable_out = e2.basis.basicIndex[2]
    candidate = TinyHiGHS.sparse_inverse_combine(e2.basis.hash, variable_out - 1)
    candidate = TinyHiGHS.sparse_combine(candidate, 0)
    push!(e2.visited_basis, candidate)
    @test is_bad_basis_change!(e2, kDual, 1, 2, TinyHiGHS.kRebuildReasonNo)
    @test length(e2.bad_basis_change) == 1 &&
          e2.bad_basis_change[1].reason == kBadBasisChangeCycling &&
          e2.bad_basis_change[1].taboo

    # Backtracking : après un INVERT réussi, corrompre la base en doublon la
    # rend singulière ; `get_nonsingular_inverse!` doit restaurer la dernière
    # base non singulière et réduire la limite d'updates de moitié.
    e3 = SimplexEngine(lp; basis=make_basis(lp, [1, 2]))
    e3.info.update_count = 8
    @test get_nonsingular_inverse!(e3, kSolvePhase2)
    @test e3.info.valid_backtracking_basis
    saved_basic = copy(e3.basis.basicIndex)
    e3.basis.basicIndex[2] = e3.basis.basicIndex[1]     # colonne dupliquée
    e3.info.update_count = 8
    @test get_nonsingular_inverse!(e3, kSolvePhase2)
    @test e3.basis.basicIndex == saved_basic
    @test e3.info.backtracking
    @test e3.info.update_limit == 4
    # Reprise après backtracking (M4c) : le solve primal doit passer par
    # `kSolvePhaseUnknown`, rétablir la phase et conclure (coûts et valeurs de
    # la base restaurée remis en place, drapeau oublié).
    initialise_for_solve!(e3)
    p = PrimalSolver(e3)
    solve!(p)
    @test e3.model_status == TinyHiGHS.kOptimal
    @test !e3.info.backtracking
    @test e3.info.primal_objective_value == 0.0
end

@testset "M3c — base initiale singulière (handleRankDeficiency)" begin
    # Deux lignes identiques : la base structurelle {1, 2} est singulière et le
    # facteur la complète par la logique de la ligne dépendante.
    A = [1.0 1.0; 1.0 1.0]
    lp = SimplexLp(2, 2, csc_from_dense(A), [1.0, 2.0], [0.0, 0.0], [10.0, 10.0],
        [1.0, 1.0], [2.0, 2.0])
    e = SimplexEngine(lp; basis=make_basis(lp, [1, 2]))
    initialise_for_solve!(e)         # ne doit pas lever
    @test any(change -> change.reason == kBadBasisChangeSingular &&
                       change.taboo, e.bad_basis_change)
    if e.model_status != TinyHiGHS.kOptimal
        d = DualSolver(e)
        solve!(d)
    end
    compute_primal_objective_value!(e)
    @test e.model_status == TinyHiGHS.kOptimal
    @test e.info.primal_objective_value == 1.0        # x1 = 1, x2 = 0
end

@testset "M3c — raffinement itératif du BTRAN unitaire" begin
    # Oracle algébrique : le résidu `Bᵀ row_ep - e_row_out` est recalculé en
    # dense depuis la matrice du LP, et le raffinement doit corriger une
    # perturbation injectée de 1e-6 sans dégrader le résidu initial.
    rng = MersenneTwister(20261017)
    checked = 0
    worst = (initial=0.0, refined=0.0)
    while checked < 30
        lp, options, A, basic = random_engine_case(rng; num_row_max=8)
        e = SimplexEngine(lp, options; basis=make_basis(lp, basic))
        set_nonbasic_move!(e)
        invert!(e.nla) == 0 || continue        # base rang-déficiente : écartée
        row_out = rand(rng, 1:lp.num_row)
        row_ep = HVector(lp.num_row)
        clear!(row_ep)
        row_ep.count = 1
        row_ep.index[1] = row_out
        row_ep.array[row_out] = 1.0
        row_ep.packFlag = true           # contrat de `HEkk::unitBtran`
        btran!(e.nla, row_ep, e.info.row_ep_density)
        # Perturbation contrôlée : le raffinement doit la retirer.
        iRow = row_ep.index[rand(rng, 1:row_ep.count)]
        row_ep.array[iRow] += 1e-6
        # Résidu dense : `[Bᵀ row_ep - e_row_out]_k` — la colonne de la
        # variable de base `basic[k]` dotée sur les lignes de la matrice (et le
        # signe +1 de la logique). `basicIndex` est celui du facteur (permuté
        # par `build!`), pas l'ordre demandé.
        basic_now = copy(e.basis.basicIndex)
        residual_max() = maximum(1:lp.num_row) do k
            iVar = basic_now[k]
            acc = -Float64(k == row_out)
            if iVar <= lp.num_col
                for r ∈ 1:lp.num_row
                    acc += A[r, iVar] * row_ep.array[r]
                end
            else
                acc += row_ep.array[iVar - lp.num_col]
            end
            abs(acc)
        end
        before = residual_max()
        TinyHiGHS.unit_btran_iterative_refinement!(e, row_out, row_ep)
        after = residual_max()
        worst = (initial=max(worst.initial, before),
            refined=max(worst.refined, after))
        checked += 1
    end
    @test checked == 30
    # La perturbation (1e-6) est visible au départ et effacée à l'arrivée.
    @test worst.initial > 1e-7
    @test worst.refined < 1e-12
    worst.refined < 1e-12 || @info "écarts BTRAN raffiné" worst
end

@testset "M3a — inférence et validation des constructeurs" begin
    @test (@inferred SimplexOptions()) isa SimplexOptions
    @test (@inferred SimplexLp(0, 0, SparseMatrix(0, 0), Float64[], Float64[],
        Float64[], Float64[], Float64[])) isa SimplexLp
    let m = SparseMatrix(2, 2)
        ensure_rowwise!(m)
        @test_throws ArgumentError SimplexLp(2, 2, m, [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])   # a_matrix non colwise
    end
    @test_throws ArgumentError SimplexLp(2, 2, SparseMatrix(2, 2), [0.0, 0.0],
        [0.0, 0.0], [0.0, 0.0], [0.0], [0.0, 0.0])        # bornes lignes
    @test_throws ArgumentError SimplexLp(2, 2, SparseMatrix(3, 2), [0.0, 0.0],
        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])   # dimensions
    @test_throws ArgumentError SimplexEngine(SimplexLp(0, 0, SparseMatrix(0, 0),
        Float64[], Float64[], Float64[], Float64[], Float64[]);
        basis=SimplexBasis(1, 0))
    lp = SimplexLp(2, 1, csc_from_dense(zeros(1, 2)), [0.0, 0.0], [0.0, 0.0],
        [1.0, 1.0], [0.0], [1.0])
    e = SimplexEngine(lp)
    set_basis!(e)
    @test invert!(e.nla) == 0
    initialise_values!(e)
    @test (@inferred set_basis!(e)) isa SimplexEngine
    @test (@inferred set_nonbasic_move!(e)) isa SimplexEngine
    @test (@inferred initialise_bound!(e, kPrimal, kSolvePhaseUnknown)) isa
          SimplexEngine
    @test (@inferred compute_primal!(e)) isa SimplexEngine
    @test (@inferred compute_dual!(e)) isa SimplexEngine
    @test (@inferred compute_primal_objective_value!(e)) isa Float64
    @test (@inferred compute_dual_objective_value!(e)) isa Float64
end
