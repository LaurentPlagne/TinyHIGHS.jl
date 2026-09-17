# Portage de `simplex/HEkkPrimal.{h,cpp}` (licence MIT, HiGHS) — tranches M4a
# (phase 2 nue), M4b (phase 1, classification infaisable/non borné) et M4c
# (poids Devex/steepest edge, rayon primal, reprise après backtracking).
#
# Porté : `initialiseSolve`, `solve` (préambule avec perturbation des bornes,
# boucle majeure, reprise `kSolvePhaseUnknown` après backtracking),
# `solvePhase1`, `solvePhase2`, `iterate`, `chuzc`/`chooseColumn`, `useVariableIn`,
# `phase1ChooseRow`, `chooseRow`, `considerBoundSwap`, `assessPivot`,
# `updateVerify`, `update`, `updateDual`, `phase1UpdatePrimal`,
# `basicFeasibilityChange*`, `phase2UpdatePrimal`, `considerInfeasibleValueIn`
# (deux phases), `adjustPerturbedEquationOut`, `rebuild`, `cleanup`,
# `correctPrimal`, `getBasicPrimalInfeasibility`, `shiftBound`,
# `getNonbasicFreeColumnSet`, `removeNonbasicFreeColumn`,
# `initialiseDevexFramework`/`updateDevex`,
# `computePrimalSteepestEdgeWeights`/`updatePrimalSteepestEdgeWeights`,
# `updateBtranPSE` et `savePrimalRay` (le rayon n'est pas exporté, §1).
#
# Non porté : le CHUZC hyper-creux, que la source gelée **désactive** dans les
# deux branches de `rebuild` (`use_hyper_chuzc = false`), la limite d'objectif
# et les rapports. Le contrôle `debugPrimalSteepestEdgeWeights` est porté pour
# ses effets de bord (RNG et FTRAN), sans son rapport. Chaque branche absente
# lève une erreur explicite plutôt que de dégrader silencieusement.
#
# `solve!` suppose l'état frais comme la source : coûts, bornes, valeurs,
# primal, dual et infaisabilités déjà calculés par l'appelant (`HEkk::solve`
# ou le nettoyage de `HEkkDual::solve`), et `INVERT` fait.

"""
    PrimalSolver(engine)

Miroir de `HEkkPrimal` : tampons `row_ep`/`row_ap`/`col_aq`, poids de colonnes
(Dantzig pour M4a) et état de l'itération courante.
"""
mutable struct PrimalSolver
    engine::SimplexEngine
    solver_num_row::Int
    solver_num_col::Int
    solver_num_tot::Int
    inv_solver_num_row::Float64
    row_ep::HVector
    row_ap::HVector
    col_aq::HVector
    col_steepest_edge::HVector
    col_basic_feasibility_change::HVector
    row_basic_feasibility_change::HVector
    ph1_sorter_r::Vector{Tuple{Float64,Int}}
    ph1_sorter_t::Vector{Tuple{Float64,Int}}
    edge_weight::Vector{Float64}
    devex_index::Vector{Int}
    num_devex_iterations::Int
    num_bad_devex_weight::Int
    nonbasic_free_col_set::Vector{Int}
    num_free_col::Int
    num_flip_since_rebuild::Int
    variable_in::Int
    move_in::Int
    row_out::Int
    variable_out::Int
    move_out::Int
    theta_dual::Float64
    theta_primal::Float64
    value_in::Float64
    alpha_col::Float64
    alpha_row::Float64
    numerical_trouble::Float64
    edge_weight_mode::Int
    solve_phase::Int
    rebuild_reason::Int
    primal_feasibility_tolerance::Float64
    dual_feasibility_tolerance::Float64
    max_max_local_primal_infeasibility::Float64
    max_max_ignored_violation::Float64
    max_max_primal_correction::Float64
end

function PrimalSolver(e::SimplexEngine)
    num_row = e.lp.num_row
    num_col = e.lp.num_col
    num_tot = num_col + num_row
    num_free_col = count(1:num_tot) do iCol
        e.info.workLower[iCol] == -kHighsInf &&
            e.info.workUpper[iCol] == kHighsInf
    end
    return PrimalSolver(e, num_row, num_col, num_tot, 1.0 / num_row,
        HVector(num_row), HVector(num_col), HVector(num_row), HVector(num_row),
        HVector(num_row), HVector(num_col), Tuple{Float64,Int}[],
        Tuple{Float64,Int}[], ones(num_tot), zeros(Int, num_tot), 0, 0, Int[],
        num_free_col, 0, 0, 0, kNoRowChosen, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, kEdgeWeightDantzig, kSolvePhase2, kRebuildReasonNo, 0.0, 0.0,
        0.0, 0.0, 0.0)
end

"""`HEkkPrimal::initialiseSolve` — poids Dantzig, Devex ou steepest edge."""
function initialise_solve!(p::PrimalSolver)
    e = p.engine
    p.primal_feasibility_tolerance = e.options.primal_feasibility_tolerance
    p.dual_feasibility_tolerance = e.options.dual_feasibility_tolerance
    e.status.has_primal_objective_value = false
    e.status.has_dual_objective_value = false
    e.model_status = kNotset
    e.solve_bailout = false
    p.rebuild_reason = kRebuildReasonNo
    if !e.status.has_dual_steepest_edge_weights
        # Pas de poids DSE duaux à maintenir : la source assigne les vecteurs
        # (utilisés autour de la factorisation et du backtracking).
        fill!(e.dual_edge_weight, 1.0)
        resize!(e.scattered_dual_edge_weight, p.solver_num_tot)
    end
    strategy = e.options.simplex_primal_edge_weight_strategy
    if strategy == kSimplexEdgeWeightStrategyChoose ||
       strategy == kSimplexEdgeWeightStrategyDevex
        # « choose » part en Devex, comme la source.
        p.edge_weight_mode = kEdgeWeightDevex
        initialise_devex_framework!(p)
    elseif strategy == kSimplexEdgeWeightStrategyDantzig
        p.edge_weight_mode = kEdgeWeightDantzig
        fill!(p.edge_weight, 1.0)
    elseif strategy == kSimplexEdgeWeightStrategySteepestEdge
        p.edge_weight_mode = kEdgeWeightSteepestEdge
        compute_primal_steepest_edge_weights!(p)
    else
        error("PrimalSolver : stratégie de poids $strategy inconnue")
    end
    return p
end

"""
`HEkkPrimal::initialiseDevexFramework` : poids à 1, ensemble de référence
`devex_index = nonbasicFlag²` (les non basiques au moment de l'initialisation),
compteurs remis à zéro.
"""
function initialise_devex_framework!(p::PrimalSolver)
    fill!(p.edge_weight, 1.0)
    fill!(p.devex_index, 0)
    for iVar ∈ 1:p.solver_num_tot
        flag = p.engine.basis.nonbasicFlag[iVar]
        p.devex_index[iVar] = flag * flag
    end
    p.num_devex_iterations = 0
    p.num_bad_devex_weight = 0
    return p
end

"""`HEkkPrimal::updateDevex` — poids Devex après le pivot."""
function update_devex!(p::PrimalSolver)
    e = p.engine
    d_pivot_weight = 0.0
    use_col_indices, to_entry = sparse_loop_style(p.col_aq.count,
        p.solver_num_row)
    for iEntry ∈ 1:to_entry
        iRow = use_col_indices ? p.col_aq.index[iEntry] : iEntry
        iCol = e.basis.basicIndex[iRow]
        d_alpha = p.devex_index[iCol] * p.col_aq.array[iRow]
        d_pivot_weight += d_alpha * d_alpha
    end
    d_pivot_weight += p.devex_index[p.variable_in] * 1.0
    # Poids aberrant : compté, la source ré-initialise au-delà du seuil.
    p.edge_weight[p.variable_in] > kBadDevexWeightFactor * d_pivot_weight &&
        (p.num_bad_devex_weight += 1)
    d_pivot = p.col_aq.array[p.row_out]
    d_pivot_weight /= d_pivot * d_pivot
    for iEl ∈ 1:p.row_ap.count
        iCol = p.row_ap.index[iEl]
        alpha = p.row_ap.array[iCol]
        devex = d_pivot_weight * alpha * alpha + p.devex_index[iCol] * 1.0
        p.edge_weight[iCol] < devex && (p.edge_weight[iCol] = devex)
    end
    for iEl ∈ 1:p.row_ep.count
        iRow = p.row_ep.index[iEl]
        iCol = iRow + p.solver_num_col
        alpha = p.row_ep.array[iRow]
        devex = d_pivot_weight * alpha * alpha + p.devex_index[iCol] * 1.0
        p.edge_weight[iCol] < devex && (p.edge_weight[iCol] = devex)
    end
    p.edge_weight[p.variable_out] = max(1.0, d_pivot_weight)
    p.edge_weight[p.variable_in] = 1.0
    p.num_devex_iterations += 1
    return p
end

"""
`HEkkPrimal::computePrimalSteepestEdgeWeights` : poids exacts. Sur base
logique, `1 + ‖a_j‖²` (somme dans l'ordre de la colonne) ; sinon un FTRAN par
variable non basique.
"""
function compute_primal_steepest_edge_weights!(p::PrimalSolver)
    e = p.engine
    resize!(p.edge_weight, p.solver_num_tot)
    if logical_basis(e)
        m = e.lp.a_matrix
        for iCol ∈ 1:p.solver_num_col
            weight = 1.0
            for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
                weight += m.value[iEl] * m.value[iEl]
            end
            p.edge_weight[iCol] = weight
        end
    else
        local_col_aq = HVector(p.solver_num_row)
        for iVar ∈ 1:p.solver_num_tot
            if e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue
                p.edge_weight[iVar] =
                    compute_primal_steepest_edge_weight(p, iVar, local_col_aq)
            end
        end
    end
    return p
end

"""`HEkkPrimal::computePrimalSteepestEdgeWeight` — un FTRAN de la colonne."""
function compute_primal_steepest_edge_weight(p::PrimalSolver, iVar::Int,
    local_col_aq::HVector)
    e = p.engine
    clear!(local_col_aq)
    collect_aj!(e.lp.a_matrix, local_col_aq, iVar, 1.0)
    local_col_aq.packFlag = false
    ftran!(e.nla, local_col_aq, e.info.col_aq_density)
    e.info.col_aq_density = update_operation_result_density(
        e.info.col_aq_density, local_col_aq.count * p.inv_solver_num_row)
    return 1 + norm2(local_col_aq)
end

"""
`HEkkPrimal::updatePrimalSteepestEdgeWeights` : mise à jour des poids par le
vecteur `mu = B^{-T} hat{a}_q` (BTRAN PSE), sur les non basiques de la rangée
pivot (structurales puis logiques).
"""
function update_primal_steepest_edge_weights!(p::PrimalSolver)
    e = p.engine
    m = e.lp.a_matrix
    copy!(p.col_steepest_edge, p.col_aq)
    update_btran_pse!(p)
    col_aq_squared_2norm = norm2(p.col_aq)
    for iX ∈ 1:(p.row_ap.count + p.row_ep.count)
        if iX <= p.row_ap.count
            iVar = p.row_ap.index[iX]
            pivotal_row_value = p.row_ap.array[iVar]
        else
            iRow = p.row_ep.index[iX - p.row_ap.count]
            iVar = p.solver_num_col + iRow
            pivotal_row_value = p.row_ep.array[iRow]
        end
        iVar == p.variable_in && continue
        e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue || continue
        lambda = pivotal_row_value / p.alpha_col
        mu_aj = 0.0
        if iVar <= p.solver_num_col
            for iEl ∈ m.start[iVar]:(m.start[iVar + 1] - 1)
                mu_aj += p.col_steepest_edge.array[m.index[iEl]] * m.value[iEl]
            end
        else
            mu_aj = p.col_steepest_edge.array[iVar - p.solver_num_col]
        end
        min_weight = 1 + lambda * lambda
        p.edge_weight[iVar] +=
            lambda * lambda * col_aq_squared_2norm - 2 * lambda * mu_aj
        p.edge_weight[iVar] += lambda * lambda
        p.edge_weight[iVar] < min_weight &&
            (p.edge_weight[iVar] = min_weight)
    end
    # La colonne tableau de la variable sortante est la colonne pivot divisée
    # par le pivot, sauf en position pivot (1/pivot).
    p.edge_weight[p.variable_out] =
        (1 + col_aq_squared_2norm) / (p.alpha_col * p.alpha_col)
    p.edge_weight[p.variable_in] = 0.0
    return p
end

"""
`HEkkPrimal::debugPrimalSteepestEdgeWeights` — branche « coûteuse » forcée par
`update` en stratégie steepest edge. La source n'imprime qu'en cas d'erreur,
mais le contrôle **consomme le RNG** (`integer(num_tot)` jusqu'à un non
basique) et lance un FTRAN par variable tirée, ce qui met à jour
`col_aq_density` : ses effets sont donc portés, sans le rapport.
"""
function check_primal_steepest_edge_weights!(p::PrimalSolver)
    e = p.engine
    num_check_weight = max(1, min(10, p.solver_num_tot ÷ 10))
    local_col_aq = HVector(p.solver_num_row)
    for _ ∈ 1:num_check_weight
        iVar = 0
        while true
            # `HighsRandom::integer` est 0-based dans la source.
            iVar = integer(e.random, p.solver_num_tot) + 1
            e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue && break
        end
        compute_primal_steepest_edge_weight(p, iVar, local_col_aq)
    end
    return p
end

"""`HEkkPrimal::updateBtranPSE` — BTRAN du vecteur des poids (densité dédiée)."""
function update_btran_pse!(p::PrimalSolver)
    e = p.engine
    btran!(e.nla, p.col_steepest_edge, e.info.col_steepest_edge_density)
    e.info.col_steepest_edge_density = update_operation_result_density(
        e.info.col_steepest_edge_density,
        p.col_steepest_edge.count * p.inv_solver_num_row)
    return p
end

"""`HEkkPrimal::getNonbasicFreeColumnSet` — ordre croissant des colonnes."""
function get_nonbasic_free_column_set!(p::PrimalSolver)
    p.num_free_col == 0 && return p
    e = p.engine
    empty!(p.nonbasic_free_col_set)
    for iCol ∈ 1:p.solver_num_tot
        if e.basis.nonbasicFlag[iCol] == kNonbasicFlagTrue &&
           e.info.workLower[iCol] <= -kHighsInf &&
           e.info.workUpper[iCol] >= kHighsInf
            push!(p.nonbasic_free_col_set, iCol)
        end
    end
    return p
end

"""
    solve!(p; force_phase2 = true)

`HEkkPrimal::solve` (M4a). `force_phase2` est `pass_force_phase2` de la
source : le nettoyage dual l'appelle à `true`. Rend le moteur ; le statut est
dans `engine.model_status`. Le nettoyage dual qui suit un `OptimalCleanup` est
câblé ; le rayon primal (non-bornitude) et la phase 1 ne sont pas portés.
"""
function solve!(p::PrimalSolver; force_phase2::Bool=false, restore::Bool=true)
    e = p.engine
    # `Highs::run` a rejeté les bornes : le LP est infaisable sans simplexe.
    e.bounds_infeasible && return e
    initialise_solve!(p)
    e.status.has_invert || error("PrimalSolver : INVERT requis avant solve")
    # `HEkk::solve` : lève d'éventuels blocages hérités d'un solve précédent et
    # oublie le rayon d'un solve antérieur.
    e.info.allow_bound_perturbation = true
    clear_ray_records!(e)
    get_nonbasic_free_column_set!(p)
    primal_feasible_with_unperturbed_bounds =
        e.info.num_primal_infeasibilities == 0
    force = force_phase2 ||
            e.info.max_primal_infeasibility^2 <
            e.options.primal_feasibility_tolerance
    no_simplex_primal_infeasibilities =
        primal_feasible_with_unperturbed_bounds || force_phase2
    near_optimal = e.info.num_dual_infeasibilities < 1000 &&
                   e.info.max_dual_infeasibility < 1e-3 &&
                   no_simplex_primal_infeasibilities
    perturb_bounds = !near_optimal
    if perturb_bounds && e.info.primal_simplex_bound_perturbation_multiplier != 0
        # Bornes écartées : valeurs et infaisabilités sont recalculées sur les
        # bornes perturbées.
        initialise_bound!(e, kPrimal, kSolvePhaseUnknown; perturb=true)
        initialise_nonbasic_value_and_move!(e)
        compute_primal!(e)
        compute_simplex_primal_infeasible!(e)
    end
    if bailout!(e)
        return_from_solve!(e, kPrimal)
        restore && restore_scale!(e)
        return e
    end
    p.solve_phase = e.info.num_primal_infeasibilities > 0 ? kSolvePhase1 :
                    kSolvePhase2
    force && (p.solve_phase = kSolvePhase2)
    while true
        it0 = e.iteration_count
        e.status.has_primal_objective_value = false
        if p.solve_phase == kSolvePhaseUnknown
            # Reprise après backtracking : le nombre d'infaisabilités primales
            # redonne la phase, et les coûts/valeurs de la base restaurée sont
            # remis en place.
            compute_simplex_primal_infeasible!(e)
            p.solve_phase = e.info.num_primal_infeasibilities > 0 ?
                            kSolvePhase1 : kSolvePhase2
            if e.info.backtracking
                initialise_cost!(e, kPrimal, p.solve_phase)
                initialise_nonbasic_value_and_move!(e)
                e.info.backtracking = false
            end
        end
        if p.solve_phase == kSolvePhase1
            solve_phase1!(p)
            e.info.primal_phase1_iteration_count += e.iteration_count - it0
        elseif p.solve_phase == kSolvePhase2
            solve_phase2!(p)
            e.info.primal_phase2_iteration_count += e.iteration_count - it0
        else
            e.model_status = kSolveError
            return e
        end
        if bailout!(e)
            return_from_solve!(e, kPrimal)
            restore && restore_scale!(e)
            return e
        end
        if p.solve_phase == kSolvePhaseTabooBasis
            e.model_status = kUnknown
            return_from_solve!(e, kPrimal)
            restore && restore_scale!(e)
            return e
        end
        if p.solve_phase == kSolvePhaseError
            e.model_status = kSolveError
            return e
        end
        # `solvePhase1` qui a nettoyé les bornes reste en phase 1 : la boucle
        # majeure reprend. `solvePhase2` peut renvoyer en phase 1 après un
        # nettoyage. La source sort sur optimal, exit et nettoyage dual.
        (p.solve_phase == kSolvePhaseOptimal ||
         p.solve_phase == kSolvePhaseExit ||
         p.solve_phase == kSolvePhaseOptimalCleanup) && break
    end
    p.solve_phase == kSolvePhaseOptimal && (e.model_status = kOptimal)
    if p.solve_phase == kSolvePhaseOptimalCleanup
        # Infaisabilités primales après phase 2 : dual faisable, donc le dual
        # nettoie (sans perturbation de coûts, stratégie duale nue).
        compute_primal_objective_value!(e)
        save_cost_perturbation =
            e.info.dual_simplex_cost_perturbation_multiplier
        e.info.dual_simplex_cost_perturbation_multiplier = 0.0
        save_strategy = e.info.simplex_strategy
        e.info.simplex_strategy = kSimplexStrategyDualPlain
        d = DualSolver(e)
        # La classification `kUnboundedOrInfeasible` appartient à `HEkk::solve`,
        # pas à l'appel imbriqué du nettoyage ; les échelles aussi.
        solve!(d; force_phase2=true, classify=false, restore=false)
        e.info.dual_simplex_cost_perturbation_multiplier =
            save_cost_perturbation
        e.info.simplex_strategy = save_strategy
    end
    return_from_solve!(e, kPrimal)
    restore && restore_scale!(e)
    return e
end

"""
    solve_phase1!(p)

`HEkkPrimal::solvePhase1` : boucle rebuild/itérations en phase 1. Rend la main
sur phase 2 (plus d'infaisabilité), sur infaisabilité prouvée
(`kInfeasible`), ou après un `cleanup` qui laisse la phase 1 à reprendre.
"""
function solve_phase1!(p::PrimalSolver)
    e = p.engine
    e.status.has_primal_objective_value = false
    e.status.has_dual_objective_value = false
    bailout!(e) && return p
    e.info.valid_backtracking_basis || put_backtracking_basis!(e)
    while true
        rebuild!(p)
        p.solve_phase == kSolvePhaseError && return p
        p.solve_phase == kSolvePhaseUnknown && return p
        bailout!(e) && return p
        # `rebuild!` a trouvé une base primalement faisable : retour phase 2.
        p.solve_phase == kSolvePhase2 && break
        while true
            iterate!(p)
            bailout!(e) && return p
            p.solve_phase == kSolvePhaseError && return p
            p.solve_phase == kSolvePhase1 || error(
                "PrimalSolver : phase $(p.solve_phase) après itération (phase 1)")
            p.rebuild_reason != kRebuildReasonNo && break
        end
        finished = e.status.has_fresh_rebuild && p.num_flip_since_rebuild == 0 &&
                   !rebuild_refactor(e, p.rebuild_reason)
        if finished && taboo_bad_basis_change(e)
            p.solve_phase = kSolvePhaseTabooBasis
            return p
        end
        finished && break
    end
    if p.solve_phase == kSolvePhase1 && p.variable_in < 0
        # Optimal en phase 1 : soit des bornes écartées cachaient la
        # faisabilité (nettoyage et reprise), soit l'infaisabilité est prouvée.
        if e.info.bounds_shifted || e.info.bounds_perturbed
            cleanup!(p)
        else
            e.model_status = kInfeasible
            p.solve_phase = kSolvePhaseExit
        end
    end
    return p
end

"""`HEkkPrimal::solvePhase2`."""
function solve_phase2!(p::PrimalSolver)
    e = p.engine
    info = e.info
    e.status.has_primal_objective_value = false
    e.status.has_dual_objective_value = false
    bailout!(e) && return p
    phase2_update_primal!(p, true)
    if !info.valid_backtracking_basis
        put_backtracking_basis!(e)
    end
    while true
        # rebuild! : singularité, ou retour en phase 1, ou sortie.
        rebuild!(p)
        p.solve_phase == kSolvePhaseError && return p
        # Backtracking : la boucle majeure reprend la main et rétablit la phase
        # (`kSolvePhaseUnknown`).
        p.solve_phase == kSolvePhaseUnknown && return p
        bailout!(e) && return p
        # `rebuild!` a trouvé une infaisabilité primale : retour en phase 1, la
        # boucle majeure reprend.
        p.solve_phase == kSolvePhase1 && break
        while true
            iterate!(p)
            bailout!(e) && return p
            p.solve_phase == kSolvePhaseError && return p
            p.solve_phase == kSolvePhase2 || error(
                "PrimalSolver : phase $(p.solve_phase) après itération")
            p.rebuild_reason != kRebuildReasonNo && break
        end
        # Données fraîches de rebuild et aucun flip : regarder ce qui s'est
        # passé avant de boucler.
        finished = e.status.has_fresh_rebuild && p.num_flip_since_rebuild == 0 &&
                   !rebuild_refactor(e, p.rebuild_reason)
        if finished && taboo_bad_basis_change(e)
            # Seul changement de base possible mais interdit : impossible de
            # conclure.
            p.solve_phase = kSolvePhaseTabooBasis
            return p
        end
        finished && break
    end
    p.solve_phase == kSolvePhase1 && return p
    if p.variable_in == -1
        # Aucun candidat CHUZC même après rebuild : probablement optimal.
        cleanup!(p)
        if e.info.num_primal_infeasibilities > 0
            p.solve_phase = kSolvePhaseOptimalCleanup
        else
            p.solve_phase = kSolvePhaseOptimal
            e.model_status = kOptimal
            compute_dual_objective_value!(e)
        end
    elseif p.row_out == kNoRowSought
        # CHUZR n'a pas eu lieu (coût réduit recalculé non attractif, sans
        # rebuild) : cas rare que la source ne traite pas non plus.
        error("PrimalSolver : row_out = kNoRowSought (cas rare non porté)")
    else
        # Aucun candidat CHUZR : primal non borné, ou retour en phase 1 après
        # nettoyage.
        if e.info.bounds_shifted || e.info.bounds_perturbed
            cleanup!(p)
            if e.info.num_primal_infeasibilities > 0
                p.solve_phase = kSolvePhase1
            end
        else
            # Non-bornitude certifiée : le rayon primal est enregistré (variable
            # entrante et signe), puis le statut est posé.
            p.solve_phase = kSolvePhaseExit
            save_primal_ray!(p)
            e.model_status = kUnbounded
        end
    end
    return p
end

"""`HEkkPrimal::iterate`."""
function iterate!(p::PrimalSolver)
    e = p.engine
    p.row_out = kNoRowSought
    chuzc!(p)
    if p.variable_in == -1
        p.rebuild_reason = kRebuildReasonPossiblyOptimal
        return p
    end
    use_variable_in!(p) || return p
    if p.solve_phase == kSolvePhase1
        phase1_choose_row!(p)
        if p.row_out == kNoRowChosen
            # Aucun candidat de pivot en phase 1 : erreur, comme la source.
            p.solve_phase = kSolvePhaseError
            return p
        end
    else
        choose_row!(p)
    end
    consider_bound_swap!(p)
    p.rebuild_reason == kRebuildReasonPossiblyPrimalUnbounded && return p
    if p.row_out >= 0
        assess_pivot!(p)
        p.rebuild_reason != kRebuildReasonNo && return p
    end
    is_bad_basis_change!(e, kPrimal, p.variable_in, p.row_out,
        p.rebuild_reason) && return p
    update!(p)
    if e.info.num_primal_infeasibilities == 0 && p.solve_phase == kSolvePhase1
        # Plus d'infaisabilité en phase 1 : forcer le rebuild qui basculera en
        # phase 2.
        p.rebuild_reason = kRebuildReasonPossiblyPhase1Feasible
    end
    ok_rebuild_reason =
        p.rebuild_reason == kRebuildReasonNo ||
        p.rebuild_reason == kRebuildReasonPossiblyPhase1Feasible ||
        p.rebuild_reason == kRebuildReasonPrimalInfeasibleInPrimalSimplex ||
        p.rebuild_reason == kRebuildReasonSyntheticClockSaysInvert ||
        p.rebuild_reason == kRebuildReasonUpdateLimitReached
    ok_rebuild_reason ||
        error("PrimalSolver : rebuild_reason $(p.rebuild_reason) inattendu")
    return p
end

"""`HEkkPrimal::chuzc` — masque les tabous, choisit la colonne entrante."""
function chuzc!(p::PrimalSolver)
    e = p.engine
    work_dual = e.info.workDual
    apply_taboo_variable_in!(e, work_dual, 0.0)
    choose_column!(p)
    unapply_taboo_variable_in!(e, work_dual)
    return p
end

"""`HEkkPrimal::chooseColumn` (Dantzig, sans hyper-creux)."""
function choose_column!(p::PrimalSolver)
    e = p.engine
    work_dual = e.info.workDual
    best_measure = 0.0
    p.variable_in = -1
    # Colonnes libres non basiques d'abord.
    for iCol ∈ p.nonbasic_free_col_set
        dual_infeasibility = abs(work_dual[iCol])
        if dual_infeasibility > p.dual_feasibility_tolerance &&
           dual_infeasibility^2 > best_measure * p.edge_weight[iCol]
            p.variable_in = iCol
            best_measure = dual_infeasibility^2 / p.edge_weight[iCol]
        end
    end
    for iCol ∈ 1:p.solver_num_tot
        dual_infeasibility = -e.basis.nonbasicMove[iCol] * work_dual[iCol]
        if dual_infeasibility > p.dual_feasibility_tolerance &&
           dual_infeasibility^2 > best_measure * p.edge_weight[iCol]
            p.variable_in = iCol
            best_measure = dual_infeasibility^2 / p.edge_weight[iCol]
        end
    end
    return p
end

"""`HEkk::pivotColumnFtran` — colonne pivot par FTRAN."""
function pivot_column_ftran!(p::PrimalSolver, iCol::Int, col_aq::HVector)
    e = p.engine
    clear!(col_aq)
    col_aq.packFlag = true
    collect_aj!(e.lp.a_matrix, col_aq, iCol, 1.0)
    ftran!(e.nla, col_aq, e.info.col_aq_density)
    e.info.col_aq_density = update_operation_result_density(
        e.info.col_aq_density, col_aq.count * p.inv_solver_num_row)
    return p
end

"""`HEkk::computeDualForTableauColumn` — dual recalculé de la colonne entrante."""
function compute_dual_for_tableau_column(p::PrimalSolver, iVar::Int,
    tableau_column::HVector)
    e = p.engine
    dual = e.info.workCost[iVar]
    for i ∈ 1:tableau_column.count
        iRow = tableau_column.index[i]
        dual -= tableau_column.array[iRow] *
                e.info.workCost[e.basis.basicIndex[iRow]]
    end
    return dual
end

"""`HEkkPrimal::useVariableIn` — FTRAN puis contrôle du dual recalculé."""
function use_variable_in!(p::PrimalSolver)
    e = p.engine
    info = e.info
    updated_theta_dual = info.workDual[p.variable_in]
    p.move_in = updated_theta_dual > 0 ? -1 : 1
    move = e.basis.nonbasicMove[p.variable_in]
    if move != 0 && move != p.move_in
        error("PrimalSolver : move_in incompatible avec nonbasicMove")
    end
    pivot_column_ftran!(p, p.variable_in, p.col_aq)
    computed_theta_dual =
        compute_dual_for_tableau_column(p, p.variable_in, p.col_aq)
    info.workDual[p.variable_in] = computed_theta_dual
    p.theta_dual = computed_theta_dual
    theta_dual_small = abs(p.theta_dual) <= p.dual_feasibility_tolerance
    theta_dual_sign_error = updated_theta_dual * computed_theta_dual <= 0
    theta_dual_small && (info.num_dual_infeasibilities -= 1)
    if theta_dual_small || theta_dual_sign_error
        if !theta_dual_small && info.update_count > 0
            p.rebuild_reason = kRebuildReasonPossiblySingularBasis
        end
        return false
    end
    return true
end

"""`HEkkPrimal::chooseRow` — ratio test (passes 1 et 2)."""
function choose_row!(p::PrimalSolver)
    e = p.engine
    info = e.info
    p.row_out = kNoRowChosen
    alpha_tol = info.update_count < 10 ? 1e-9 :
                info.update_count < 20 ? 1e-8 : 1e-7
    # Passe 1 : plus petit theta relâché (tolérance d'infaisabilité).
    relax_theta = 1e100
    for i ∈ 1:p.col_aq.count
        iRow = p.col_aq.index[i]
        alpha = p.col_aq.array[iRow] * p.move_in
        if alpha > alpha_tol
            relax_space = info.baseValue[iRow] - info.baseLower[iRow] +
                          p.primal_feasibility_tolerance
            if relax_space < relax_theta * alpha
                relax_theta = relax_space / alpha
            end
        elseif alpha < -alpha_tol
            relax_space = info.baseValue[iRow] - info.baseUpper[iRow] -
                          p.primal_feasibility_tolerance
            if relax_space > relax_theta * alpha
                relax_theta = relax_space / alpha
            end
        end
    end
    # Passe 2 : plus grand |alpha| parmi les lignes à theta relâché.
    best_alpha = 0.0
    for i ∈ 1:p.col_aq.count
        iRow = p.col_aq.index[i]
        alpha = p.col_aq.array[iRow] * p.move_in
        if alpha > alpha_tol
            tight_space = info.baseValue[iRow] - info.baseLower[iRow]
            if tight_space < relax_theta * alpha && best_alpha < alpha
                best_alpha = alpha
                p.row_out = iRow
            end
        elseif alpha < -alpha_tol
            tight_space = info.baseValue[iRow] - info.baseUpper[iRow]
            if tight_space > relax_theta * alpha && best_alpha < -alpha
                best_alpha = -alpha
                p.row_out = iRow
            end
        end
    end
    return p
end

"""
    phase1_choose_row!(p)

`HEkkPrimal::phase1ChooseRow` : ratio test de phase 1. Les candidats sont les
thetas où la pente de l'infaisabilité primale change (`ph1_sorter_r`, theta
relâché par la tolérance) et ceux où une infaisabilité est résorbée ou créée
(`ph1_sorter_t`, theta serré). Le theta retenu est le dernier avant que la
pente ne devienne négative ; le pivot est le plus grand `|alpha|` parmi les
candidats serrés sous ce theta, à 10 % du maximum.

Le marqueur signé reprend le couple C++ `(theta, iRow)`, `(theta, iRow - m)` :
positif = candidat de la borne supérieure, négatif = borne inférieure, encodé
`iRow - m - 1` en 1-based pour que l'ordre de tri (et les égalités) soit
identique à la source.
"""
function phase1_choose_row!(p::PrimalSolver)
    e = p.engine
    info = e.info
    alpha_tol = info.update_count < 10 ? 1e-9 :
                info.update_count < 20 ? 1e-8 : 1e-7
    empty!(p.ph1_sorter_r)
    empty!(p.ph1_sorter_t)
    for i ∈ 1:p.col_aq.count
        iRow = p.col_aq.index[i]
        alpha = p.col_aq.array[iRow] * p.move_in
        if alpha > alpha_tol
            # La variable de base décroît.
            if info.baseValue[iRow] >
               info.baseUpper[iRow] + p.primal_feasibility_tolerance
                # Elle devient faisable en atteignant sa borne supérieure.
                feas_theta = (info.baseValue[iRow] - info.baseUpper[iRow] -
                              p.primal_feasibility_tolerance) / alpha
                push!(p.ph1_sorter_r, (feas_theta, iRow))
                push!(p.ph1_sorter_t, (feas_theta, iRow))
            end
            if info.baseValue[iRow] >
               info.baseLower[iRow] - p.primal_feasibility_tolerance &&
               info.baseLower[iRow] > -kHighsInf
                # Elle redevient infaisable en passant sous sa borne inférieure.
                relax_theta = (info.baseValue[iRow] - info.baseLower[iRow] +
                               p.primal_feasibility_tolerance) / alpha
                tight_theta = (info.baseValue[iRow] - info.baseLower[iRow]) / alpha
                push!(p.ph1_sorter_r,
                    (relax_theta, iRow - p.solver_num_row - 1))
                push!(p.ph1_sorter_t,
                    (tight_theta, iRow - p.solver_num_row - 1))
            end
        end
        if alpha < -alpha_tol
            # La variable de base croît.
            if info.baseValue[iRow] <
               info.baseLower[iRow] - p.primal_feasibility_tolerance
                feas_theta = (info.baseValue[iRow] - info.baseLower[iRow] +
                              p.primal_feasibility_tolerance) / alpha
                push!(p.ph1_sorter_r,
                    (feas_theta, iRow - p.solver_num_row - 1))
                push!(p.ph1_sorter_t,
                    (feas_theta, iRow - p.solver_num_row - 1))
            end
            if info.baseValue[iRow] <
               info.baseUpper[iRow] + p.primal_feasibility_tolerance &&
               info.baseUpper[iRow] < kHighsInf
                relax_theta = (info.baseValue[iRow] - info.baseUpper[iRow] -
                               p.primal_feasibility_tolerance) / alpha
                tight_theta = (info.baseValue[iRow] - info.baseUpper[iRow]) / alpha
                push!(p.ph1_sorter_r, (relax_theta, iRow))
                push!(p.ph1_sorter_t, (tight_theta, iRow))
            end
        end
    end
    if isempty(p.ph1_sorter_r)
        p.row_out = kNoRowChosen
        p.variable_out = -1
        return p
    end
    sort!(p.ph1_sorter_r)
    max_theta = p.ph1_sorter_r[1][1]
    gradient = abs(p.theta_dual)
    for (my_theta, marker) in p.ph1_sorter_r
        iRow = marker > 0 ? marker : marker + p.solver_num_row + 1
        gradient -= abs(p.col_aq.array[iRow])
        gradient <= 0 && break
        max_theta = my_theta
    end
    sort!(p.ph1_sorter_t)
    max_alpha = 0.0
    # `i_last` est l'indice (1-based) du premier theta trop grand ; il est
    # exclu du parcours arrière, comme le `iLast` 0-based de la source.
    i_last = length(p.ph1_sorter_t) + 1
    for i ∈ 1:length(p.ph1_sorter_t)
        my_theta, marker = p.ph1_sorter_t[i]
        iRow = marker > 0 ? marker : marker + p.solver_num_row + 1
        abs_alpha = abs(p.col_aq.array[iRow])
        if my_theta > max_theta
            i_last = i
            break
        end
        max_alpha = max(max_alpha, abs_alpha)
    end
    p.row_out = kNoRowChosen
    p.variable_out = -1
    p.move_out = 0
    for i ∈ (i_last - 1):-1:1
        marker = p.ph1_sorter_t[i][2]
        iRow = marker > 0 ? marker : marker + p.solver_num_row + 1
        if abs(p.col_aq.array[iRow]) > max_alpha * 0.1
            p.row_out = iRow
            p.move_out = marker > 0 ? 1 : -1
            break
        end
    end
    return p
end

"""`HEkkPrimal::considerBoundSwap` — theta primal, flip ou pivot."""
function consider_bound_swap!(p::PrimalSolver)
    e = p.engine
    info = e.info
    if p.row_out == kNoRowChosen
        # Aucune ligne bloquante : flip ou non borné.
        p.theta_primal = p.move_in * kHighsInf
        p.move_out = 0
    else
        p.alpha_col = p.col_aq.array[p.row_out]
        # En phase 1, `move_out` vient de `phase1ChooseRow` : la variable
        # sortante peut devenir faisable (vers sa borne) ou le rester, la
        # direction n'est pas déductible du signe du pivot.
        p.solve_phase == kSolvePhase2 &&
            (p.move_out = p.alpha_col * p.move_in > 0 ? -1 : 1)
        p.theta_primal = p.move_out == 1 ?
                         (info.baseValue[p.row_out] -
                          info.baseUpper[p.row_out]) / p.alpha_col :
                         (info.baseValue[p.row_out] -
                          info.baseLower[p.row_out]) / p.alpha_col
    end
    flipped = false
    lower_in = info.workLower[p.variable_in]
    upper_in = info.workUpper[p.variable_in]
    p.value_in = info.workValue[p.variable_in] + p.theta_primal
    if p.move_in > 0
        if p.value_in > upper_in + p.primal_feasibility_tolerance
            flipped = true
            p.row_out = kNoRowChosen
            p.value_in = upper_in
            p.theta_primal = upper_in - lower_in
        end
    else
        if p.value_in < lower_in - p.primal_feasibility_tolerance
            flipped = true
            p.row_out = kNoRowChosen
            p.value_in = lower_in
            p.theta_primal = lower_in - upper_in
        end
    end
    if p.solve_phase == kSolvePhase2 && !(p.row_out >= 0 || flipped)
        # Non-bornitude possible : en phase 1, row_out >= 0 est garanti (son
        # absence est traitée comme une erreur dans `iterate!`).
        p.rebuild_reason = kRebuildReasonPossiblyPrimalUnbounded
    end
    return p
end

"""`HEkk::unitBtran` — BTRAN du vecteur unité e_row_out."""
function unit_btran!(p::PrimalSolver, iRow::Int, row_ep::HVector)
    e = p.engine
    clear!(row_ep)
    row_ep.count = 1
    row_ep.index[1] = iRow
    row_ep.array[iRow] = 1.0
    row_ep.packFlag = true
    btran!(e.nla, row_ep, e.info.row_ep_density)
    e.info.row_ep_density = update_operation_result_density(
        e.info.row_ep_density, row_ep.count * p.inv_solver_num_row)
    return p
end

"""`HEkkPrimal::assessPivot` — BTRAN unitaire, PRICE, contrôle numérique."""
function assess_pivot!(p::PrimalSolver)
    e = p.engine
    p.alpha_col = p.col_aq.array[p.row_out]
    p.variable_out = e.basis.basicIndex[p.row_out]
    unit_btran!(p, p.row_out, p.row_ep)
    tableau_row_price!(e, p.row_ep, p.row_ap)
    update_verify!(p)
    return p
end

"""`HEkkPrimal::updateVerify` — pivot colonne contre pivot rangée."""
function update_verify!(p::PrimalSolver)
    e = p.engine
    p.numerical_trouble = 0.0
    abs_alpha_from_col = abs(p.alpha_col)
    p.alpha_row = p.variable_in <= p.solver_num_col ?
                  p.row_ap.array[p.variable_in] :
                  p.row_ep.array[p.variable_in - p.solver_num_col]
    abs_alpha_from_row = abs(p.alpha_row)
    min_abs_alpha = min(abs_alpha_from_col, abs_alpha_from_row)
    p.numerical_trouble =
        abs(abs_alpha_from_col - abs_alpha_from_row) / min_abs_alpha
    if p.numerical_trouble > 1e-7 && e.info.update_count > 0
        p.rebuild_reason = kRebuildReasonPossiblySingularBasis
    end
    return p
end

"""`HEkkPrimal::adjustPerturbedEquationOut` — sortie d'une variable fixe."""
function adjust_perturbed_equation_out!(p::PrimalSolver)
    e = p.engine
    info = e.info
    info.bounds_perturbed || return p
    lp = e.lp
    if p.variable_out <= p.solver_num_col
        lp_lower = lp.col_lower[p.variable_out]
        lp_upper = lp.col_upper[p.variable_out]
    else
        iRow = p.variable_out - p.solver_num_col
        lp_lower = -lp.row_upper[iRow]
        lp_upper = -lp.row_lower[iRow]
    end
    lp_lower < lp_upper && return p
    # La variable sortante est fixe : theta primal rectifié sur sa vraie valeur.
    true_fixed_value = lp_lower
    p.theta_primal =
        (info.baseValue[p.row_out] - true_fixed_value) / p.alpha_col
    info.workLower[p.variable_out] = true_fixed_value
    info.workUpper[p.variable_out] = true_fixed_value
    info.workRange[p.variable_out] = 0.0
    p.value_in = info.workValue[p.variable_in] + p.theta_primal
    return p
end

"""`HEkkPrimal::update` — mise à jour primale, duale, facteur et pivots."""
function update!(p::PrimalSolver)
    e = p.engine
    info = e.info
    flipped = p.row_out < 0
    if flipped
        p.variable_out = p.variable_in
        p.alpha_col = 0.0
        p.numerical_trouble = 0.0
        info.workValue[p.variable_in] = p.value_in
        e.basis.nonbasicMove[p.variable_in] == p.move_in ||
            error("PrimalSolver : flip incohérent")
        e.basis.nonbasicMove[p.variable_in] = -p.move_in
    else
        adjust_perturbed_equation_out!(p)
    end
    # Copie de l'ordre de la source : mise à jour primale, puis duale. En phase
    # 1, les duals sont corrigés des changements de faisabilité (le CHUZC
    # hyper-creux, non porté, s'insère ici).
    if p.solve_phase == kSolvePhase1
        phase1_update_primal!(p)
        basic_feasibility_change_update_dual!(p)
    else
        phase2_update_primal!(p, false)
    end
    if flipped
        info.primal_bound_swap += 1
        invalidate_dual_infeasibility_record!(e)
        p.num_flip_since_rebuild += 1
        e.total_synthetic_tick += p.col_aq.synthetic_tick
        return p
    end
    info.baseValue[p.row_out] = p.value_in
    consider_infeasible_value_in!(p)
    p.theta_dual = info.workDual[p.variable_in]
    update_dual!(p)
    if p.edge_weight_mode == kEdgeWeightDevex
        update_devex!(p)
    elseif p.edge_weight_mode == kEdgeWeightSteepestEdge
        # Le contrôle « coûteux » de la source (avant mise à jour) est porté
        # pour ses effets de bord : tirages du RNG et FTRAN de contrôle.
        check_primal_steepest_edge_weights!(p)
        update_primal_steepest_edge_weights!(p)
    end
    remove_nonbasic_free_column!(p)
    if e.status.has_dual_steepest_edge_weights
        # Le primal doit maintenir les poids DSE duaux : le dual peut reprendre
        # la main (nettoyage `OptimalCleanup`) et les utiliser.
        update_dual_steepest_edge_weights!(p)
    end
    transform_for_update!(e.nla, p.col_aq, p.row_ep, p.variable_in, p.row_out)
    update_pivots!(e, p.variable_in, p.row_out, p.move_out)
    p.rebuild_reason =
        update_factor!(e, p.col_aq, p.row_ep, p.row_out, p.rebuild_reason)
    p.edge_weight_mode == kEdgeWeightSteepestEdge &&
        check_primal_steepest_edge_weights!(p)
    update_matrix!(e, p.variable_in, p.variable_out)
    info.update_count >= info.update_limit &&
        (p.rebuild_reason = kRebuildReasonUpdateLimitReached)
    e.iteration_count += 1
    # Trop de poids Devex aberrants : ré-initialiser l'ensemble de référence.
    p.edge_weight_mode == kEdgeWeightDevex &&
        p.num_bad_devex_weight > kAllowedNumBadDevexWeight &&
        initialise_devex_framework!(p)
    e.total_synthetic_tick += p.col_aq.synthetic_tick
    e.total_synthetic_tick += p.row_ep.synthetic_tick
    return p
end

"""`HEkkPrimal::updateDual` — duals après le pas de primal simplex."""
function update_dual!(p::PrimalSolver)
    e = p.engine
    info = e.info
    p.theta_dual = info.workDual[p.variable_in] / p.alpha_col
    for i ∈ 1:p.row_ap.count
        iCol = p.row_ap.index[i]
        info.workDual[iCol] -= p.theta_dual * p.row_ap.array[iCol]
    end
    for i ∈ 1:p.row_ep.count
        iRow = p.row_ep.index[i]
        iCol = p.solver_num_col + iRow
        info.workDual[iCol] -= p.theta_dual * p.row_ep.array[iRow]
    end
    info.workDual[p.variable_in] = 0.0
    info.workDual[p.variable_out] = -p.theta_dual
    invalidate_dual_infeasibility_record!(e)
    e.status.has_dual_objective_value = false
    return p
end

"""
    phase1_compute_dual!(p)

`HEkkPrimal::phase1ComputeDual` : coûts et duals de phase 1. `workCost` vaut
`±1` sur les variables de base infaisables (écarté par le multiplicateur de
perturbation de phase 1, fixé à 1), puis `workDual = -[A I]' B^{-T} workCost`
restreint aux non basiques — un BTRAN et un PRICE complets.
"""
function phase1_compute_dual!(p::PrimalSolver)
    e = p.engine
    info = e.info
    buffer = e.dual_col
    clear!(buffer)
    fill!(info.workCost, 0.0)
    fill!(info.workDual, 0.0)
    base = info.primal_simplex_phase1_cost_perturbation_multiplier * 5e-7
    for iRow ∈ 1:p.solver_num_row
        value = info.baseValue[iRow]
        lower = info.baseLower[iRow]
        upper = info.baseUpper[iRow]
        bound_violated = value < lower - p.primal_feasibility_tolerance ? -1 :
                         value > upper + p.primal_feasibility_tolerance ? 1 : 0
        bound_violated == 0 && continue
        cost = Float64(bound_violated)
        base != 0.0 && (cost *= 1 + base * info.numTotRandomValue[iRow])
        buffer.array[iRow] = cost
        buffer.count += 1
        buffer.index[buffer.count] = iRow
    end
    buffer.count > 0 ||
        error("PrimalSolver : phase 1 sans infaisabilité (second membre nul)")
    for iRow ∈ 1:p.solver_num_row
        info.workCost[e.basis.basicIndex[iRow]] = buffer.array[iRow]
    end
    full_btran!(e, buffer)
    buffer_long = e.dual_row
    full_price!(e, buffer, buffer_long)
    for iCol ∈ 1:p.solver_num_col
        info.workDual[iCol] =
            -e.basis.nonbasicFlag[iCol] * buffer_long.array[iCol]
    end
    for iRow ∈ 1:p.solver_num_row
        iCol = p.solver_num_col + iRow
        info.workDual[iCol] = -e.basis.nonbasicFlag[iCol] * buffer.array[iRow]
    end
    return p
end

"""
`HEkkPrimal::phase1UpdatePrimal` — valeurs de base et coûts de phase 1 ; les
changements de coût des basiques sont collectés dans
`col_basic_feasibility_change` pour la correction des duals.
"""
function phase1_update_primal!(p::PrimalSolver)
    e = p.engine
    info = e.info
    change = p.col_basic_feasibility_change
    clear!(change)
    base = info.primal_simplex_phase1_cost_perturbation_multiplier * 5e-7
    for iEl ∈ 1:p.col_aq.count
        iRow = p.col_aq.index[iEl]
        info.baseValue[iRow] -= p.theta_primal * p.col_aq.array[iRow]
        iCol = e.basis.basicIndex[iRow]
        was_cost = info.workCost[iCol]
        value = info.baseValue[iRow]
        lower = info.baseLower[iRow]
        upper = info.baseUpper[iRow]
        bound_violated = value < lower - p.primal_feasibility_tolerance ? -1 :
                         value > upper + p.primal_feasibility_tolerance ? 1 : 0
        cost = Float64(bound_violated)
        base != 0.0 && (cost *= 1 + base * info.numTotRandomValue[iRow])
        info.workCost[iCol] = cost
        if was_cost != 0.0
            cost == 0.0 && (info.num_primal_infeasibilities -= 1)
        elseif cost != 0.0
            info.num_primal_infeasibilities += 1
        end
        delta_cost = cost - was_cost
        if delta_cost != 0.0
            change.array[iRow] = delta_cost
            change.count += 1
            change.index[change.count] = iRow
            # Les coûts des logiques n'ont pas de composante dans le PRICE :
            # leur delta est appliqué au dual directement.
            iCol > p.solver_num_col && (info.workDual[iCol] += delta_cost)
        end
    end
    invalidate_primal_max_sum_infeasibility_record!(e)
    return p
end

"""
`HEkkPrimal::basicFeasibilityChangeUpdateDual` : BTRAN du vecteur des
changements de coût des basiques, PRICE sur les non basiques, puis soustraction
des composantes — les logiques reçoivent le vecteur BTRAN (leur coût est
exactement la variable logique).
"""
function basic_feasibility_change_update_dual!(p::PrimalSolver)
    e = p.engine
    info = e.info
    basic_feasibility_change_btran!(p)
    basic_feasibility_change_price!(p)
    use_row_indices, to_entry = sparse_loop_style(
        p.row_basic_feasibility_change.count, p.solver_num_col)
    for iEntry ∈ 1:to_entry
        iCol = use_row_indices ? p.row_basic_feasibility_change.index[iEntry] :
               iEntry
        info.workDual[iCol] -= p.row_basic_feasibility_change.array[iCol]
    end
    use_col_indices, to_entry = sparse_loop_style(
        p.col_basic_feasibility_change.count, p.solver_num_row)
    for iEntry ∈ 1:to_entry
        iRow = use_col_indices ? p.col_basic_feasibility_change.index[iEntry] :
               iEntry
        info.workDual[p.solver_num_col + iRow] -=
            p.col_basic_feasibility_change.array[iRow]
    end
    invalidate_dual_infeasibility_record!(e)
    return p
end

"""`HEkkPrimal::basicFeasibilityChangeBtran` — BTRAN du vecteur de changement."""
function basic_feasibility_change_btran!(p::PrimalSolver)
    e = p.engine
    btran!(e.nla, p.col_basic_feasibility_change,
        e.info.col_basic_feasibility_change_density)
    e.info.col_basic_feasibility_change_density = update_operation_result_density(
        e.info.col_basic_feasibility_change_density,
        p.col_basic_feasibility_change.count * p.inv_solver_num_row)
    return p
end

"""`HEkkPrimal::basicFeasibilityChangePrice` — PRICE du vecteur de changement."""
function basic_feasibility_change_price!(p::PrimalSolver)
    e = p.engine
    local_density = p.col_basic_feasibility_change.count / p.solver_num_row
    use_col_price, use_row_price_w_switch = choose_price_technique(e,
        local_density)
    clear!(p.row_basic_feasibility_change)
    if use_col_price
        price_by_column!(e.lp.a_matrix, p.row_basic_feasibility_change,
            p.col_basic_feasibility_change)
        # Le PRICE colonne couvre aussi les basiques : les annuler via le
        # masque non basique.
        for iCol ∈ 1:p.solver_num_col
            p.row_basic_feasibility_change.array[iCol] *=
                e.basis.nonbasicFlag[iCol]
        end
    elseif use_row_price_w_switch
        price_by_row_with_switch!(e.ar_matrix, p.row_basic_feasibility_change,
            p.col_basic_feasibility_change,
            e.info.row_basic_feasibility_change_density, 1, kHyperPriceDensity)
    else
        price_by_row!(e.ar_matrix, p.row_basic_feasibility_change,
            p.col_basic_feasibility_change)
    end
    e.info.row_basic_feasibility_change_density = update_operation_result_density(
        e.info.row_basic_feasibility_change_density,
        p.row_basic_feasibility_change.count / p.solver_num_col)
    return p
end

"""
`HEkkPrimal::phase2UpdatePrimal` — valeurs de base, infaisabilités traitées par
shift de bornes (stratégie `Always` de la source), objectif mis à jour.
"""
function phase2_update_primal!(p::PrimalSolver, initialise::Bool)
    e = p.engine
    info = e.info
    if initialise
        p.max_max_local_primal_infeasibility = 0.0
        p.max_max_ignored_violation = 0.0
        return p
    end
    use_col_indices, to_entry =
        sparse_loop_style(p.col_aq.count, p.solver_num_row)
    for iEntry ∈ 1:to_entry
        iRow = use_col_indices ? p.col_aq.index[iEntry] : iEntry
        info.baseValue[iRow] -= p.theta_primal * p.col_aq.array[iRow]
        lower = info.baseLower[iRow]
        upper = info.baseUpper[iRow]
        value = info.baseValue[iRow]
        bound_violated = value < lower - p.primal_feasibility_tolerance ? -1 :
                         value > upper + p.primal_feasibility_tolerance ? 1 :
                         0
        bound_violated == 0 && continue
        # Stratégie `Always` : le dépassement est absorbé par un shift de borne.
        iCol = e.basis.basicIndex[iRow]
        if bound_violated > 0
            bound, shift = shift_bound!(p, false, iCol, value,
                info.numTotRandomValue[iCol])
            info.workUpper[iCol] = bound
            info.baseUpper[iRow] = bound
            info.workUpperShift[iCol] += shift
        else
            bound, shift = shift_bound!(p, true, iCol, value,
                info.numTotRandomValue[iCol])
            info.workLower[iCol] = bound
            info.baseLower[iRow] = bound
            info.workLowerShift[iCol] += shift
        end
        info.bounds_shifted = true
    end
    info.updated_primal_objective_value +=
        info.workDual[p.variable_in] * p.theta_primal
    return p
end

"""`HEkkPrimal::considerInfeasibleValueIn` — valeur entrante infaisable."""
function consider_infeasible_value_in!(p::PrimalSolver)
    e = p.engine
    info = e.info
    lower = info.workLower[p.variable_in]
    upper = info.workUpper[p.variable_in]
    bound_violated =
        p.value_in < lower - p.primal_feasibility_tolerance ? -1 :
        p.value_in > upper + p.primal_feasibility_tolerance ? 1 : 0
    bound_violated == 0 && return p
    if p.solve_phase == kSolvePhase1
        # Phase 1 : la valeur entrante infaisable devient un coût de base
        # (perturbé par le multiplicateur de phase 1) et bascule le dual.
        info.num_primal_infeasibilities += 1
        base = info.primal_simplex_phase1_cost_perturbation_multiplier * 5e-7
        cost = Float64(bound_violated)
        base != 0.0 && (cost *= 1 + base * info.numTotRandomValue[p.row_out])
        info.workCost[p.variable_in] = cost
        info.workDual[p.variable_in] += cost
        invalidate_primal_max_sum_infeasibility_record!(e)
        return p
    end
    # Phase 2, stratégie `Always` : shift de borne.
    if bound_violated > 0
        bound, shift = shift_bound!(p, false, p.variable_in, p.value_in,
            info.numTotRandomValue[p.variable_in])
        info.workUpper[p.variable_in] = bound
        info.workUpperShift[p.variable_in] += shift
    else
        bound, shift = shift_bound!(p, true, p.variable_in, p.value_in,
            info.numTotRandomValue[p.variable_in])
        info.workLower[p.variable_in] = bound
        info.workLowerShift[p.variable_in] += shift
    end
    info.bounds_perturbed = true
    invalidate_primal_max_sum_infeasibility_record!(e)
    return p
end

"""`HEkkPrimal::removeNonbasicFreeColumn` — la colonne entrante devient basique."""
function remove_nonbasic_free_column!(p::PrimalSolver)
    e = p.engine
    e.basis.nonbasicMove[p.variable_in] == 0 || return p
    idx = findfirst(==(p.variable_in), p.nonbasic_free_col_set)
    idx === nothing &&
        error("PrimalSolver : colonne libre non basique absente de l'ensemble")
    deleteat!(p.nonbasic_free_col_set, idx)
    return p
end

"""
`HEkkPrimal::updateDualSteepestEdgeWeights` — maintien des poids DSE duaux
pendant une phase primale (le dual peut reprendre la main au nettoyage).
"""
function update_dual_steepest_edge_weights!(p::PrimalSolver)
    e = p.engine
    copy!(p.col_steepest_edge, p.row_ep)
    # `updateFtranDSE` : retrait de l'échelle de ligne puis FTRAN en espace
    # échelonné.
    unapply_basis_matrix_row_scale!(e.nla, p.col_steepest_edge)
    ftran_in_scaled_space!(e.nla, p.col_steepest_edge, e.info.row_DSE_density)
    e.info.row_DSE_density = update_operation_result_density(
        e.info.row_DSE_density, p.col_steepest_edge.count * p.inv_solver_num_row)
    edge_weight = e.dual_edge_weight
    # `simplex_in_scaled_space_` : norme de `row_ep` directement.
    edge_weight[p.row_out] = e.lp.is_scaled ? norm2(p.row_ep) :
                             row_ep_2norm_in_scaled_space(e.nla, p.row_out,
                                 p.row_ep)
    pivot_in_scaled_space_value =
        pivot_in_scaled_space(e.nla, p.col_aq, p.variable_in, p.row_out)
    new_pivotal_edge_weight = edge_weight[p.row_out] /
                              (pivot_in_scaled_space_value^2)
    kai = -2 / pivot_in_scaled_space_value
    update_dual_steepest_edge_weights!(e, p.row_out, p.variable_in, p.col_aq,
        new_pivotal_edge_weight, kai, p.col_steepest_edge.array)
    edge_weight[p.row_out] = new_pivotal_edge_weight
    return p
end

"""`HEkkPrimal::shiftBound` — borne et décalage pour rendre `value` faisable."""
function shift_bound!(p::PrimalSolver, lower::Bool, iVar::Int,
    value::Float64, random_value::Float64)
    feasibility = (1 + random_value) * p.primal_feasibility_tolerance
    if lower
        value < p.engine.info.workLower[iVar] - p.primal_feasibility_tolerance ||
            error("PrimalSolver : shiftBound lower sans violation")
        infeasibility = p.engine.info.workLower[iVar] - value
        shift = infeasibility + feasibility
        bound = p.engine.info.workLower[iVar] - shift
    else
        value > p.engine.info.workUpper[iVar] + p.primal_feasibility_tolerance ||
            error("PrimalSolver : shiftBound upper sans violation")
        infeasibility = value - p.engine.info.workUpper[iVar]
        shift = infeasibility + feasibility
        bound = p.engine.info.workUpper[iVar] + shift
    end
    return bound, shift
end

"""
`HEkkPrimal::savePrimalRay` : variable entrante et signe opposé à son
mouvement. Le vecteur du rayon (export) reste hors périmètre (§1).
"""
function save_primal_ray!(p::PrimalSolver)
    p.variable_in >= 0 ||
        error("PrimalSolver : rayon primal sans variable entrante")
    p.move_in != kNoRaySign || error("PrimalSolver : rayon primal sans signe")
    ray = p.engine.primal_ray_record
    clear!(ray)
    ray.index = p.variable_in
    ray.sign = -p.move_in
    return p
end

"""`HEkkPrimal::getBasicPrimalInfeasibility` — num/max/sum sur les basiques."""
function get_basic_primal_infeasibility!(p::PrimalSolver)
    info = p.engine.info
    tolerance = p.primal_feasibility_tolerance
    info.num_primal_infeasibilities = 0
    info.max_primal_infeasibility = 0.0
    info.sum_primal_infeasibilities = 0.0
    for iRow ∈ 1:p.solver_num_row
        value = info.baseValue[iRow]
        lower = info.baseLower[iRow]
        upper = info.baseUpper[iRow]
        primal_infeasibility = 0.0
        if value < lower - tolerance
            primal_infeasibility = lower - value
        elseif value > upper + tolerance
            primal_infeasibility = value - upper
        end
        if primal_infeasibility > 0
            primal_infeasibility > tolerance &&
                (info.num_primal_infeasibilities += 1)
            info.max_primal_infeasibility =
                max(primal_infeasibility, info.max_primal_infeasibility)
            info.sum_primal_infeasibilities += primal_infeasibility
        end
    end
    return p
end

"""`HEkkPrimal::rebuild` — INVERT, primal, phase, duals et objectif."""
function rebuild!(p::PrimalSolver)
    e = p.engine
    info = e.info
    status = e.status
    clear_bad_basis_change_taboo_flag!(e)
    check_updated_objective_value = status.has_primal_objective_value
    previous_primal_objective_value =
        check_updated_objective_value ? info.updated_primal_objective_value :
        -kHighsInf
    refactor_basis_matrix = rebuild_refactor(e, p.rebuild_reason)
    p.rebuild_reason = kRebuildReasonNo
    if refactor_basis_matrix
        if !get_nonsingular_inverse!(e, p.solve_phase)
            p.solve_phase = kSolvePhaseError
            return p
        end
        reset_synthetic_clock!(e)
    end
    if !status.has_ar_matrix
        # N'arrive qu'en backtracking.
        info.backtracking || error("PrimalSolver : ar_matrix absente")
        initialise_partitioned_rowwise_matrix!(e)
    end
    if info.backtracking
        # Le backtracking peut changer de phase : on ressort.
        p.solve_phase = kSolvePhaseUnknown
        return p
    end
    compute_primal!(e)
    p.solve_phase == kSolvePhase2 && correct_primal!(p)
    get_basic_primal_infeasibility!(p)
    if info.num_primal_infeasibilities > 0
        # Infaisabilités primales : la source bascule en phase 1 et recalcule
        # coûts et duals de phase 1.
        p.solve_phase == kSolvePhase2 && (p.solve_phase = kSolvePhase1)
        phase1_compute_dual!(p)
    else
        if p.solve_phase == kSolvePhase1
            initialise_cost!(e, kPrimal, p.solve_phase)
            p.solve_phase = kSolvePhase2
        end
        compute_dual!(e)
    end
    compute_simplex_dual_infeasible!(e)
    compute_primal_objective_value!(e)
    if check_updated_objective_value
        correction = info.primal_objective_value - previous_primal_objective_value
        info.updated_primal_objective_value += correction
    end
    info.updated_primal_objective_value = info.primal_objective_value
    p.num_flip_since_rebuild = 0
    status.has_fresh_rebuild = true
    return p
end

"""`HEkkPrimal::correctPrimal` — shifts sur les infaisabilités du rebuild."""
function correct_primal!(p::PrimalSolver, initialise::Bool=false)
    e = p.engine
    info = e.info
    if initialise
        p.max_max_primal_correction = 0.0
        return true
    end
    p.solve_phase == kSolvePhase2 ||
        error("PrimalSolver : correctPrimal hors phase 2")
    num_primal_correction = 0
    max_primal_correction = 0.0
    sum_primal_correction = 0.0
    num_primal_correction_skipped = 0
    for iRow ∈ 1:p.solver_num_row
        lower = info.baseLower[iRow]
        upper = info.baseUpper[iRow]
        value = info.baseValue[iRow]
        bound_violated = value < lower - p.primal_feasibility_tolerance ? -1 :
                         value > upper + p.primal_feasibility_tolerance ? 1 :
                         0
        bound_violated == 0 && continue
        if info.allow_bound_perturbation
            iCol = e.basis.basicIndex[iRow]
            shift = 0.0
            if bound_violated > 0
                bound, shift = shift_bound!(p, false, iCol, value,
                    info.numTotRandomValue[iCol])
                info.workUpper[iCol] = bound
                info.baseUpper[iRow] = bound
                info.workUpperShift[iCol] += shift
            else
                bound, shift = shift_bound!(p, true, iCol, value,
                    info.numTotRandomValue[iCol])
                info.workLower[iCol] = bound
                info.baseLower[iRow] = bound
                info.workLowerShift[iCol] += shift
            end
            num_primal_correction += 1
            max_primal_correction = max(shift, max_primal_correction)
            sum_primal_correction += shift
            info.bounds_perturbed = true
        else
            num_primal_correction_skipped += 1
        end
    end
    num_primal_correction_skipped > 0 && return false
    if max_primal_correction > 2 * p.max_max_primal_correction
        p.max_max_primal_correction = max_primal_correction
    end
    return true
end

"""`HEkkPrimal::cleanup` — retire perturbations et shifts de bornes."""
function cleanup!(p::PrimalSolver)
    e = p.engine
    info = e.info
    (info.bounds_shifted || info.bounds_perturbed) || return p
    initialise_bound!(e, kPrimal, p.solve_phase; perturb=false)
    initialise_nonbasic_value_and_move!(e)
    info.allow_bound_perturbation = false
    compute_primal!(e)
    compute_simplex_primal_infeasible!(e)
    compute_primal_objective_value!(e)
    info.updated_primal_objective_value = info.primal_objective_value
    compute_simplex_dual_infeasible!(e)
    return p
end
