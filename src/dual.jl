# Portage de `simplex/HEkkDual.{h,cpp}` (licence MIT, HiGHS) — M3b/M3c :
# itération duale, DSE/Devex et bascule, perturbation des coûts, taboos,
# backtracking et cycle de ré-inversion.
#
# Porté : `initialiseSolve`, `solvePhase1`, `solvePhase2`, `rebuild`, `cleanup`
# (sans nettoyage primal), `iterate`, `chooseRow` (Dantzig), `chooseColumn`
# (ratio test `DualRow`, `improveChooseColumnRow`), `updateFtran`/
# `updateFtranBFRT`, `updateVerify`, `updateDual`, `updatePrimal`,
# `updatePivots`, `correctDualInfeasibilities`,
# `computeDualInfeasibilitiesWithFixedVariableFlips`,
# `assessPhase1Optimality`, `exitPhase1ResetDuals`,
# `assessPossiblyDualUnbounded` (avec preuve d'infaisabilité).
#
# Non porté : multi-pivot/PAMI (`chooseColumnSlice`, `HEkkDualMulti`), rayons,
# rapports. Chaque branche absente lève une erreur explicite plutôt que de
# dégrader silencieusement.

"""
    DualSolver(engine)

Miroir de `HEkkDual` : tampons `row_ep`/`row_ap`/`col_aq`/`col_BFRT`, état de
l'itération courante et politique de prix.
"""
mutable struct DualSolver
    engine::SimplexEngine
    solver_num_row::Int
    solver_num_col::Int
    solver_num_tot::Int
    inv_solver_num_row::Float64
    row_ep::HVector
    row_ap::HVector
    col_aq::HVector
    col_BFRT::HVector
    row_out::Int
    variable_out::Int
    move_out::Int
    variable_in::Int
    delta_primal::Float64
    theta_dual::Float64
    theta_primal::Float64
    alpha_col::Float64
    alpha_row::Float64
    numerical_trouble::Float64
    computed_edge_weight::Float64
    edge_weight_mode::Int
    num_devex_iterations::Int
    new_devex_framework::Bool
    solve_phase::Int
    rebuild_reason::Int
    force_phase2::Bool
    dual_infeas_count::Int
    initial_basis_is_logical::Bool
    dual_row::DualRow
    dual_rhs::DualRHS
end

function DualSolver(e::SimplexEngine)
    num_row = e.lp.num_row
    num_col = e.lp.num_col
    return DualSolver(e, num_row, num_col, num_col + num_row,
        1.0 / num_row, HVector(num_row), HVector(num_col), HVector(num_row),
        HVector(num_row), kNoRowChosen, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, kEdgeWeightDantzig, 0, false, kSolvePhase2, kRebuildReasonNo,
        false, 0, true, DualRow(e), DualRHS(e))
end

"""
    reset!(d, e = d.engine)

Réinitialise intégralement un `DualSolver` existant pour une nouvelle résolution.
L'état obtenu est bit-à-bit strictement identique à un `DualSolver(e)` fraîchement
construit, avec zéro allocation.
"""
function reset!(d::DualSolver, e::SimplexEngine=d.engine)
    num_row = e.lp.num_row
    num_col = e.lp.num_col
    num_tot = num_col + num_row
    d.engine = e
    d.solver_num_row = num_row
    d.solver_num_col = num_col
    d.solver_num_tot = num_tot
    d.inv_solver_num_row = 1.0 / num_row
    reset!(d.row_ep)
    reset!(d.row_ap)
    reset!(d.col_aq)
    reset!(d.col_BFRT)
    d.row_out = kNoRowChosen
    d.variable_out = 0
    d.move_out = 0
    d.variable_in = 0
    d.delta_primal = 0.0
    d.theta_dual = 0.0
    d.theta_primal = 0.0
    d.alpha_col = 0.0
    d.alpha_row = 0.0
    d.numerical_trouble = 0.0
    d.computed_edge_weight = 0.0
    d.edge_weight_mode = kEdgeWeightDantzig
    d.num_devex_iterations = 0
    d.new_devex_framework = false
    d.solve_phase = kSolvePhase2
    d.rebuild_reason = kRebuildReasonNo
    d.force_phase2 = false
    d.dual_infeas_count = 0
    d.initial_basis_is_logical = true
    reset!(d.dual_row, e)
    reset!(d.dual_rhs, e)
    return d
end

"""`HEkkDual::initialiseSolve`."""
function initialise_solve!(d::DualSolver)
    e = d.engine
    d.initial_basis_is_logical = true
    for iRow ∈ 1:d.solver_num_row
        if e.basis.basicIndex[iRow] <= d.solver_num_col
            d.initial_basis_is_logical = false
            break
        end
    end
    strategy = e.info.dual_edge_weight_strategy
    if strategy == kSimplexEdgeWeightStrategyChoose
        d.edge_weight_mode = kEdgeWeightSteepestEdge
        e.info.allow_dual_steepest_edge_to_devex_switch = true
    elseif strategy == kSimplexEdgeWeightStrategyDantzig
        d.edge_weight_mode = kEdgeWeightDantzig
        e.info.allow_dual_steepest_edge_to_devex_switch = false
    elseif strategy == kSimplexEdgeWeightStrategySteepestEdge
        d.edge_weight_mode = kEdgeWeightSteepestEdge
        e.info.allow_dual_steepest_edge_to_devex_switch = false
    elseif strategy == kSimplexEdgeWeightStrategyDevex
        d.edge_weight_mode = kEdgeWeightDevex
        e.info.allow_dual_steepest_edge_to_devex_switch = false
    else
        error("DualSolver : stratégie de poids $strategy inconnue")
    end
    e.model_status = kNotset
    e.solve_bailout = false
    d.rebuild_reason = kRebuildReasonNo
    d.dual_infeas_count = 0
    return d
end

"""
    solve!(d; force_phase2 = false, classify = true)

`HEkkDual::solve` (sous-ensemble), suivi de la classification
`kUnboundedOrInfeasible` de `HEkk::solve` (le primal tranche) quand `classify`.
Suppose le facteur frais (`compute_factor!`) et `set_basis!` faits. Rend le
moteur ; le statut est dans `engine.model_status`. `classify = false` pour
l'appel imbriqué du nettoyage primal, où la classification n'a pas lieu
(elle appartient au niveau `HEkk::solve`).
"""
function solve!(d::DualSolver; force_phase2::Bool=false, classify::Bool=true,
    restore::Bool=true)
    e = d.engine
    # `Highs::run` a rejeté les bornes : le LP est infaisable sans simplexe.
    e.bounds_infeasible && return e
    initialise_solve!(d)
    initialise_control!(e)
    # `HEkk::solve` : oublie le rayon d'un solve antérieur.
    clear_ray_records!(e)
    d.solver_num_row > 0 || error("DualSolver : LP sans ligne")
    e.status.has_invert || error("DualSolver : INVERT requis avant solve")
    # `HEkk::solve` : lève d'éventuels blocages hérités d'un solve précédent.
    e.info.allow_cost_shifting = true
    e.info.allow_cost_perturbation = true
    e.info.allow_bound_perturbation = true
    # Duals avec coûts non perturbés.
    initialise_cost!(e, kDual, kSolvePhaseUnknown)
    compute_dual!(e)
    compute_simplex_dual_infeasible!(e)
    dual_feasible_with_unperturbed_costs =
        e.info.num_dual_infeasibilities == 0
    d.force_phase2 = force_phase2 ||
                    e.info.max_dual_infeasibility^2 <
                    e.options.dual_feasibility_tolerance
    no_simplex_dual_infeasibilities =
        dual_feasible_with_unperturbed_costs || d.force_phase2
    near_optimal = no_simplex_dual_infeasibilities &&
                   e.info.num_primal_infeasibilities < 1000 &&
                   e.info.max_primal_infeasibility < 1e-3
    perturb_costs = !near_optimal
    initialise_cost!(e, kDual, kSolvePhaseUnknown; perturb=perturb_costs)
    bailout!(e) && return e
    # Poids de rangée (`HEkkDual::solve`) : unitaires en Dantzig et pour une
    # base logique ; DSE exact sinon.
    if !e.status.has_dual_steepest_edge_weights
        fill!(e.dual_edge_weight, 1.0)
        resize!(e.scattered_dual_edge_weight, d.solver_num_tot)
        if d.edge_weight_mode == kEdgeWeightSteepestEdge
            if d.initial_basis_is_logical
                e.status.has_dual_steepest_edge_weights = true
            elseif near_optimal
                # Base non logique quasi optimale : Devex plutôt que DSE.
                d.edge_weight_mode = kEdgeWeightDevex
            else
                compute_dual_steepest_edge_weights!(e, true)
                e.status.has_dual_steepest_edge_weights = true
            end
        end
        if d.edge_weight_mode == kEdgeWeightDevex
            initialise_devex_framework!(d)
        end
    end
    if perturb_costs
        compute_dual!(e)
        compute_dual_infeasibilities_with_fixed_variable_flips!(d)
        d.dual_infeas_count = e.info.num_dual_infeasibilities
    end
    if d.force_phase2
        d.solve_phase = kSolvePhase2
    else
        d.solve_phase = d.dual_infeas_count > 0 ? kSolvePhase1 : kSolvePhase2
    end
    e.solve_start_time = time()
    while d.solve_phase != kSolvePhaseOptimal
        it0 = e.iteration_count
        e.status.has_dual_objective_value = false
        if d.solve_phase == kSolvePhaseUnknown
            initialise_bound!(e, kDual, kSolvePhaseUnknown)
            initialise_nonbasic_value_and_move!(e)
            compute_dual_infeasibilities_with_fixed_variable_flips!(d)
            d.dual_infeas_count = e.info.num_dual_infeasibilities
            d.solve_phase = d.dual_infeas_count > 0 ? kSolvePhase1 :
                            kSolvePhase2
            if e.info.backtracking
                # Bounds et valeurs pour la phase retenue, puis on oublie le
                # backtracking.
                initialise_bound!(e, kDual, d.solve_phase)
                initialise_nonbasic_value_and_move!(e)
                e.info.backtracking = false
            end
        end
        if d.solve_phase == kSolvePhaseTabooBasis
            e.model_status = kUnknown
            return_from_solve!(e, kDual)
            return e
        end
        if d.solve_phase == kSolvePhase1
            solve_phase1!(d)
            e.info.dual_phase1_iteration_count += e.iteration_count - it0
        elseif d.solve_phase == kSolvePhase2
            solve_phase2!(d)
            e.info.dual_phase2_iteration_count += e.iteration_count - it0
        else
            e.model_status = kSolveError
            return e
        end
        e.solve_bailout && return return_from_solve!(e, kDual)
        if d.solve_phase == kSolvePhaseError
            e.model_status = kSolveError
            return e
        end
        (d.solve_phase == kSolvePhaseExit ||
         d.solve_phase == kSolvePhaseOptimalCleanup ||
         d.solve_phase == kSolvePhasePrimalInfeasibleCleanup) && break
    end
    # `HEkkDual::solve` : après la boucle, nettoyage primal des infaisabilités
    # duales résiduelles (M4a).
    if d.solve_phase == kSolvePhaseOptimalCleanup ||
       d.solve_phase == kSolvePhasePrimalInfeasibleCleanup
        e.dual_simplex_cleanup_level += 1
        if d.solve_phase == kSolvePhasePrimalInfeasibleCleanup
            # Infaisabilités inconnues après une fausse non-bornitude : les
            # calculer avant le nettoyage.
            compute_simplex_infeasible!(e)
        end
        if e.dual_simplex_cleanup_level >
           e.options.max_dual_simplex_cleanup_level
            e.model_status = d.solve_phase == kSolvePhaseOptimalCleanup ?
                             kOptimal : kInfeasible
        else
            save = e.info.primal_simplex_bound_perturbation_multiplier
            e.info.primal_simplex_bound_perturbation_multiplier = 0
            p = PrimalSolver(e)
            # Appel imbriqué : le niveau « HApp » rendra les échelles.
            solve!(p; force_phase2=true, restore=false)
            e.info.primal_simplex_bound_perturbation_multiplier = save
            e.solve_bailout && return return_from_solve!(e, kDual)
            if e.model_status == kOptimal &&
               e.info.num_primal_infeasibilities +
               e.info.num_dual_infeasibilities > 0
                # La source avertit seulement : optimal malgré des
                # infaisabilités tolérées.
            end
        end
    end
    # `HEkkDual::solve` rend un état normalisé (`HEkk::returnFromSolve`), puis
    # le niveau « HApp » retire les échelles (`moveBackLpAndUnapplyScaling`).
    return_from_solve!(e, kDual)
    restore && restore_scale!(e)
    # `HEkk::solve` : le dual peut conclure `kUnboundedOrInfeasible` faute de
    # preuve d'infaisabilité primale ; le primal tranche (M4b). HiGHS ne
    # l'évite que si `allow_unbounded_or_infeasible` est demandé (défaut faux).
    if classify && e.model_status == kUnboundedOrInfeasible
        p = PrimalSolver(e)
        solve!(p)
    end
    return e
end

"""
    solve_phase1!(d)

`HEkkDual::solvePhase1` : bornes de phase 1, boucle rebuild/itérations, puis
arrêt de phase (phase 2, infaisabilité duale ou erreur).
"""
function solve_phase1!(d::DualSolver)
    e = d.engine
    e.status.has_primal_objective_value = false
    e.status.has_dual_objective_value = false
    d.rebuild_reason = kRebuildReasonNo
    bailout!(e) && return d
    initialise_bound!(e, kDual, d.solve_phase)
    initialise_nonbasic_value_and_move!(e)
    e.info.valid_backtracking_basis || put_backtracking_basis!(e)
    while true
        rebuild!(d)
        if d.solve_phase == kSolvePhaseError
            e.model_status = kSolveError
            return d
        end
        d.solve_phase == kSolvePhaseUnknown && return d
        bailout!(e) && break
        while true
            iterate!(d)
            bailout!(e) && break
            d.rebuild_reason != kRebuildReasonNo && break
        end
        e.solve_bailout && break
        finished = e.status.has_fresh_rebuild &&
                   !rebuild_refactor(e, d.rebuild_reason)
        if finished && taboo_bad_basis_change(e)
            d.solve_phase = kSolvePhaseTabooBasis
            return d
        end
        finished && break
    end
    e.solve_bailout && return d
    if d.row_out == kNoRowChosen
        if e.info.dual_objective_value == 0
            d.solve_phase = kSolvePhase2
        else
            assess_phase1_optimality!(d)
        end
    elseif d.rebuild_reason == kRebuildReasonChooseColumnFail ||
           d.rebuild_reason == kRebuildReasonExcessivePrimalValue
        d.solve_phase = kSolvePhaseError
        e.model_status = kSolveError
    elseif d.variable_in == -1
        # Phase 1 duale non bornée. Avec des coûts perturbés, on retire la
        # perturbation et on poursuit si plus aucune infaisabilité duale.
        if e.info.costs_perturbed
            cleanup!(d)
            d.dual_infeas_count == 0 && (d.solve_phase = kSolvePhase2)
        else
            d.solve_phase = kSolvePhaseError
            e.model_status = kSolveError
        end
    end
    if d.solve_phase == kSolvePhase2 || d.solve_phase == kSolvePhaseExit ||
       d.solve_phase == kSolvePhaseError
        initialise_bound!(e, kDual, kSolvePhase2)
        initialise_nonbasic_value_and_move!(e)
        if d.solve_phase == kSolvePhase2 &&
           e.dual_simplex_phase1_cleanup_level <
           e.options.max_dual_simplex_phase1_cleanup_level
            e.info.allow_cost_shifting = true
            e.info.allow_cost_perturbation = true
        end
    end
    return d
end

"""
    solve_phase2!(d)

`HEkkDual::solvePhase2` : boucle de phase 2, puis nettoyage des perturbations
(ici sans effet) et conclusion (optimal, infaisable, retour en phase 1).
"""
function solve_phase2!(d::DualSolver)
    e = d.engine
    e.status.has_primal_objective_value = false
    e.status.has_dual_objective_value = false
    d.rebuild_reason = kRebuildReasonNo
    d.solve_phase = kSolvePhase2
    e.solve_bailout = false
    bailout!(e) && return d
    create_freelist!(d.dual_row)
    e.info.valid_backtracking_basis || put_backtracking_basis!(e)
    while true
        rebuild!(d)
        if d.solve_phase == kSolvePhaseError
            e.model_status = kSolveError
            return d
        end
        d.solve_phase == kSolvePhaseUnknown && return d
        bailout!(e) && break
        bailout_on_dual_objective!(d) && break
        d.dual_infeas_count > 0 && break
        while true
            iterate!(d)
            bailout!(e) && break
            bailout_on_dual_objective!(d) && break
            if d.rebuild_reason == kRebuildReasonPossiblyDualUnbounded
                assess_possibly_dual_unbounded!(d)
            end
            d.rebuild_reason != kRebuildReasonNo && break
        end
        e.solve_bailout && break
        finished = e.status.has_fresh_rebuild &&
                   !rebuild_refactor(e, d.rebuild_reason)
        if finished && taboo_bad_basis_change(e)
            d.solve_phase = kSolvePhaseTabooBasis
            return d
        end
        finished && break
    end
    e.solve_bailout && return d
    if d.dual_infeas_count > 0
        d.solve_phase = kSolvePhase1
    elseif d.row_out == kNoRowChosen
        cleanup!(d)
        if d.dual_infeas_count > 0
            d.solve_phase = kSolvePhaseOptimalCleanup
        else
            d.solve_phase = kSolvePhaseOptimal
            e.model_status = kOptimal
        end
    elseif d.rebuild_reason == kRebuildReasonChooseColumnFail ||
           d.rebuild_reason == kRebuildReasonExcessivePrimalValue
        d.solve_phase = kSolvePhaseError
        e.model_status = kSolveError
    else
        d.solve_phase = kSolvePhaseExit
        e.model_status = kInfeasible
    end
    return d
end

"""`HEkkDual::rebuild` — duals, correction des infaisabilités, primals, liste."""
function rebuild!(d::DualSolver)
    e = d.engine
    clear_bad_basis_change_taboo_flag!(e)
    refactor_basis_matrix = rebuild_refactor(e, d.rebuild_reason)
    local_rebuild_reason = d.rebuild_reason
    d.rebuild_reason = kRebuildReasonNo
    if refactor_basis_matrix
        if !get_nonsingular_inverse!(e, d.solve_phase)
            d.solve_phase = kSolvePhaseError
            return d
        end
        reset_synthetic_clock!(e)
    end
    e.status.has_ar_matrix || initialise_partitioned_rowwise_matrix!(e)
    check_updated_objective_value = e.status.has_dual_objective_value
    previous_dual_objective_value = e.info.updated_dual_objective_value
    compute_dual!(e)
    if e.info.backtracking
        # Reprise après restauration : la phase sera redéterminée.
        d.solve_phase = kSolvePhaseUnknown
        return d
    end
    correct_dual_infeasibilities!(d)
    compute_primal!(e)
    create_array_of_primal_infeasibilities!(d.dual_rhs)
    create_infeas_list!(d.dual_rhs, e.info.col_aq_density)
    compute_dual_objective_value!(e, d.solve_phase)
    if check_updated_objective_value
        e.info.updated_dual_objective_value +=
            e.info.dual_objective_value - previous_dual_objective_value
    end
    e.info.updated_dual_objective_value = e.info.dual_objective_value
    reset_synthetic_clock!(e)
    invalidate_primal_infeasibility_record!(e)
    invalidate_dual_infeasibility_record!(e)
    e.status.has_fresh_rebuild = true
    return d
end

"""`HEkkDual::cleanup` — coûts non perturbés, duals et infaisabilités."""
function cleanup!(d::DualSolver)
    e = d.engine
    if d.solve_phase == kSolvePhase1
        # La source journalise puis `assert` au-delà de
        # `max_dual_simplex_phase1_cleanup_level` : en `NDEBUG` (oracle et JLL
        # de production) elle **poursuit**. Le port levait une erreur — il
        # arrêtait le solve là où HiGHS continue, ce qui faisait échouer le
        # rejeu échelonné là où le niveau est dépassé après divergence. Le
        # niveau reste incrémenté : il garde `allow_cost_perturbation`
        # (`solve_phase1!`).
        e.dual_simplex_phase1_cleanup_level += 1
    end
    initialise_cost!(e, kDual, kSolvePhaseUnknown)
    e.info.allow_cost_perturbation = false
    initialise_bound!(e, kDual, d.solve_phase)
    compute_dual!(e)
    compute_simplex_dual_infeasible!(e)
    d.dual_infeas_count = e.info.num_dual_infeasibilities
    compute_dual_objective_value!(e, d.solve_phase)
    e.info.updated_dual_objective_value = e.info.dual_objective_value
    return d
end

"""`HEkkDual::iterate` — une itération du simplexe dual."""
function iterate!(d::DualSolver)
    choose_row!(d)
    choose_column!(d)
    # `HEkkDual::isBadBasisChange` : changement mauvais ou cyclage.
    is_bad_basis_change!(d.engine, kDual, d.variable_in, d.row_out,
        d.rebuild_reason) && return d
    update_ftran_bfrt!(d)
    update_ftran!(d)
    if d.edge_weight_mode == kEdgeWeightSteepestEdge
        update_ftran_dse!(d)
    end
    update_verify!(d)
    update_dual!(d)
    d.engine.status.has_primal_objective_value = false
    update_primal!(d)
    update_pivots!(d)
    if d.new_devex_framework
        initialise_devex_framework!(d)
    end
    # `HEkkDual::iterationAnalysis` : contrôle de bascule DSE → Devex.
    if d.edge_weight_mode == kEdgeWeightSteepestEdge && switch_to_devex!(d)
        d.edge_weight_mode = kEdgeWeightDevex
        initialise_devex_framework!(d)
    end
    return d
end

"""`HEkkDual::chooseRow` (Dantzig) — rangée sortante et BTRAN de son e_p."""
function choose_row!(d::DualSolver)
    d.rebuild_reason != kRebuildReasonNo && return d
    e = d.engine
    apply_taboo_row_out!(e, d.dual_rhs.work_infeasibility, 0.0)
    while true
        d.row_out = choose_normal!(d.dual_rhs)
        if d.row_out == kNoRowChosen
            # La source ne restaure pas ici : `rebuild` recalculera les
            # infaisabilités.
            d.rebuild_reason = kRebuildReasonPossiblyOptimal
            return d
        end
        clear!(d.row_ep)
        d.row_ep.count = 1
        d.row_ep.index[1] = d.row_out
        d.row_ep.array[d.row_out] = 1.0
        d.row_ep.packFlag = true
        btran!(e.nla, d.row_ep, e.info.row_ep_density)
        if d.edge_weight_mode == kEdgeWeightSteepestEdge
            # Contrôle du poids DSE : on recalcule le poids exact et on
            # n'accepte la rangée que s'il n'est pas trop sous-estimé.
            updated_edge_weight = e.dual_edge_weight[d.row_out]
            # `simplex_in_scaled_space_` : le facteur est déjà dans l'espace
            # échelonné, la norme exacte est celle de `row_ep`.
            d.computed_edge_weight = e.lp.is_scaled ? norm2(d.row_ep) :
                                     row_ep_2norm_in_scaled_space(e.nla,
                                         d.row_out, d.row_ep)
            e.dual_edge_weight[d.row_out] = d.computed_edge_weight
            accept_dual_steepest_edge_weight!(d, updated_edge_weight) && break
        else
            break                       # Dantzig : acceptation immédiate
        end
    end
    unapply_taboo_row_out!(e, d.dual_rhs.work_infeasibility)
    d.variable_out = e.basis.basicIndex[d.row_out]
    if e.info.baseValue[d.row_out] < e.info.baseLower[d.row_out]
        d.delta_primal = e.info.baseValue[d.row_out] -
                         e.info.baseLower[d.row_out]
    else
        d.delta_primal = e.info.baseValue[d.row_out] -
                         e.info.baseUpper[d.row_out]
    end
    d.move_out = d.delta_primal < 0 ? -1 : 1
    e.info.row_ep_density = update_operation_result_density(
        e.info.row_ep_density, d.row_ep.count * d.inv_solver_num_row)
    return d
end

"""
`HEkkDual::chooseColumn` — prix de la rangée pivot et ratio test dual. Quand la
première passe choisit un pivot trop petit (sous `dual_simplex_pivot_growth_tolerance`),
`improve_choose_column_row!` raffine la rangée et CHUZC reprend ; les passes
suivantes retirent le pivot du paquet et recommencent.
"""
function choose_column!(d::DualSolver)
    d.rebuild_reason != kRebuildReasonNo && return d
    e = d.engine
    tableau_row_price!(e, d.row_ep, d.row_ap)
    dual_row = d.dual_row
    clear!(dual_row)
    dual_row.workDelta = d.delta_primal
    create_freemove!(dual_row, d.row_ep)
    choose_makepack!(dual_row, d.row_ap, 0)
    choose_makepack!(dual_row, d.row_ep, d.solver_num_col)
    row_ep_scale = get_value_scale(dual_row.packCount, dual_row.packValue)
    chuzc_pass = 0
    while true
        choose_possible!(dual_row)
        d.variable_in = -1
        if dual_row.workTheta <= 0 || dual_row.workCount == 0
            d.rebuild_reason = kRebuildReasonPossiblyDualUnbounded
            return d
        end
        if choose_final!(dual_row) != 0
            d.rebuild_reason = kRebuildReasonChooseColumnFail
            return d
        end
        if dual_row.workPivot >= 0
            growth_tolerance = e.options.dual_simplex_pivot_growth_tolerance
            scaled_value = row_ep_scale * dual_row.workAlpha
            if abs(scaled_value) <= growth_tolerance
                if chuzc_pass == 0
                    # Premier échec : tenter une rangée pivot plus précise.
                    improve_choose_column_row!(d)
                else
                    # Pivot retiré du paquet ; CHUZC recommence sur le reste.
                    for i ∈ 1:dual_row.packCount
                        if dual_row.packIndex[i] == dual_row.workPivot
                            dual_row.packIndex[i] =
                                dual_row.packIndex[dual_row.packCount]
                            dual_row.packValue[i] =
                                dual_row.packValue[dual_row.packCount]
                            dual_row.packCount -= 1
                            break
                        end
                    end
                end
                # Aucun pivot choisi pour cette passe.
                dual_row.workPivot = -1
            end
        else
            break
        end
        (dual_row.workPivot >= 0 || dual_row.packCount <= 0) && break
        chuzc_pass += 1
    end
    delete_freemove!(dual_row)
    d.variable_in = dual_row.workPivot
    d.alpha_row = dual_row.workAlpha
    d.theta_dual = dual_row.workTheta
    if d.edge_weight_mode == kEdgeWeightDevex && !d.new_devex_framework
        compute_devex_weight!(dual_row)
        d.computed_edge_weight = max(1.0, dual_row.computed_edge_weight)
    end
    return d
end

"""
    improve_choose_column_row!(d)

`HEkkDual::improveChooseColumnRow` : raffine la rangée pivot (`row_ep`) quand
le premier choix de CHUZC échoue sur un pivot trop petit. Le BTRAN unitaire est
raffiné itérativement (résidu en double-double), le prix est refait en
double-double (`HighsCDouble`), puis les sections 0/1 de CHUZC sont
ré-exécutées : le mouvement provisoire des colonnes libres est annulé puis
recalculé sur la rangée raffinée, et le paquet `row_ap`/`row_ep` est reconstruit.
La boucle de `choose_column!` reprend ensuite en passe 1, avec l'échelle de
pivot d'origine (`row_ep_scale` de la première passe, comme la source).
"""
function improve_choose_column_row!(d::DualSolver)
    e = d.engine
    dual_row = d.dual_row
    delete_freemove!(dual_row)
    unit_btran_iterative_refinement!(e, d.row_out, d.row_ep)
    tableau_row_price!(e, d.row_ep, d.row_ap, true)
    clear!(dual_row)
    dual_row.workDelta = d.delta_primal
    create_freemove!(dual_row, d.row_ep)
    choose_makepack!(dual_row, d.row_ap, 0)
    choose_makepack!(dual_row, d.row_ep, d.solver_num_col)
    return d
end

"""`HEkkDual::updateFtranBFRT` — flips retournés au bord, puis FTRAN du BFRT."""
function update_ftran_bfrt!(d::DualSolver)
    d.rebuild_reason != kRebuildReasonNo && return d
    e = d.engine
    update_flip!(d.dual_row, d.col_BFRT)
    if d.col_BFRT.count != 0
        ftran!(e.nla, d.col_BFRT, e.info.col_BFRT_density)
    end
    e.info.col_BFRT_density = update_operation_result_density(
        e.info.col_BFRT_density, d.col_BFRT.count * d.inv_solver_num_row)
    return d
end

"""`HEkkDual::updateFtran` — colonne pivot par FTRAN."""
function update_ftran!(d::DualSolver)
    d.rebuild_reason != kRebuildReasonNo && return d
    e = d.engine
    clear!(d.col_aq)
    d.col_aq.packFlag = true
    collect_aj!(e.lp.a_matrix, d.col_aq, d.variable_in, 1.0)
    ftran!(e.nla, d.col_aq, e.info.col_aq_density)
    e.info.col_aq_density = update_operation_result_density(
        e.info.col_aq_density, d.col_aq.count * d.inv_solver_num_row)
    d.alpha_col = d.col_aq.array[d.row_out]
    return d
end

"""`HEkkDual::updateVerify` — pivots colonne/rangée et ré-inversion."""
function update_verify!(d::DualSolver)
    d.rebuild_reason != kRebuildReasonNo && return d
    reinvert, measure = reinvert_on_numerical_trouble!(d.engine, d.alpha_col,
        d.alpha_row, kNumericalTroubleTolerance)
    d.numerical_trouble = measure
    reinvert && (d.rebuild_reason = kRebuildReasonPossiblySingularBasis)
    return d
end

"""`HEkkDual::updateDual` — duals après le pas, avec shifts si theta = 0."""
function update_dual!(d::DualSolver)
    d.rebuild_reason != kRebuildReasonNo && return d
    e = d.engine
    info = e.info
    if d.theta_dual == 0
        shift_cost!(d, d.variable_in, -info.workDual[d.variable_in])
    else
        update_dual!(d.dual_row, d.theta_dual)
    end
    variable_in_delta_dual = info.workDual[d.variable_in]
    variable_in_value = info.workValue[d.variable_in]
    variable_in_nonbasic_flag = e.basis.nonbasicFlag[d.variable_in]
    dual_objective_value_change = variable_in_nonbasic_flag *
        (-variable_in_value * variable_in_delta_dual) * e.cost_scale
    info.updated_dual_objective_value += dual_objective_value_change
    info.workDual[d.variable_in] = 0.0
    info.workDual[d.variable_out] = -d.theta_dual
    shift_back!(d, d.variable_out)
    return d
end

"""`HEkkDual::updatePrimal` — primals (et poids, absents en Dantzig)."""
function update_primal!(d::DualSolver)
    d.rebuild_reason != kRebuildReasonNo && return d
    e = d.engine
    info = e.info
    if d.edge_weight_mode == kEdgeWeightDevex
        updated_edge_weight = e.dual_edge_weight[d.row_out]
        e.dual_edge_weight[d.row_out] = d.computed_edge_weight
        d.new_devex_framework =
            new_devex_framework_needed(d, updated_edge_weight)
    end
    ok = update_primal!(d.dual_rhs, d.col_BFRT, 1.0)
    update_infeas_list!(d.dual_rhs, d.col_BFRT)
    !ok && (d.rebuild_reason = kRebuildReasonExcessivePrimalValue; return d)
    x_out = info.baseValue[d.row_out]
    l_out = info.baseLower[d.row_out]
    u_out = info.baseUpper[d.row_out]
    d.theta_primal = (x_out - (d.delta_primal < 0 ? l_out : u_out)) / d.alpha_col
    if !update_primal!(d.dual_rhs, d.col_aq, d.theta_primal)
        d.rebuild_reason = kRebuildReasonExcessivePrimalValue
        return d
    end
    update_bad_basis_change!(e, d.col_aq, d.theta_primal)
    if d.edge_weight_mode == kEdgeWeightSteepestEdge
        pivot_in_scaled_space_value =
            pivot_in_scaled_space(e.nla, d.col_aq, d.variable_in, d.row_out)
        new_pivotal_edge_weight = e.dual_edge_weight[d.row_out] /
                                  (pivot_in_scaled_space_value *
                                   pivot_in_scaled_space_value)
        kai = -2 / pivot_in_scaled_space_value
        update_dual_steepest_edge_weights!(e, d.row_out, d.variable_in, d.col_aq,
            new_pivotal_edge_weight, kai, d.row_ep.array)
        e.dual_edge_weight[d.row_out] = new_pivotal_edge_weight
    elseif d.edge_weight_mode == kEdgeWeightDevex
        new_pivotal_edge_weight = e.dual_edge_weight[d.row_out] /
                                  (d.alpha_col * d.alpha_col)
        new_pivotal_edge_weight = max(1.0, new_pivotal_edge_weight)
        update_dual_devex_weights!(e, d.col_aq, new_pivotal_edge_weight)
        e.dual_edge_weight[d.row_out] = new_pivotal_edge_weight
        d.num_devex_iterations += 1
    end
    update_infeas_list!(d.dual_rhs, d.col_aq)
    e.total_synthetic_tick += d.col_aq.synthetic_tick + d.row_ep.synthetic_tick
    return d
end

"""`HEkkDual::updateFtranDSE` — FTRAN de `row_ep` pour le poids DSE."""
function update_ftran_dse!(d::DualSolver)
    d.rebuild_reason != kRebuildReasonNo && return d
    e = d.engine
    unapply_basis_matrix_row_scale!(e.nla, d.row_ep)
    ftran_in_scaled_space!(e.nla, d.row_ep, e.info.row_DSE_density)
    e.info.row_DSE_density = update_operation_result_density(
        e.info.row_DSE_density, d.row_ep.count * d.inv_solver_num_row)
    return d
end

"""
`HEkkDual::initialiseDevexFramework` : ensemble de référence = variables
basiques (`devex_index = 1 - nonbasicFlag²`), poids remis à 1.
"""
function initialise_devex_framework!(d::DualSolver)
    e = d.engine
    @inbounds @simd for vr ∈ 1:d.solver_num_tot
        flag = Int(e.basis.nonbasicFlag[vr])
        e.info.devex_index[vr] = 1 - flag * flag
    end
    fill!(e.dual_edge_weight, 1.0)
    d.num_devex_iterations = 0
    d.new_devex_framework = false
    return d
end

"""`HEkkDual::newDevexFramework` — une nouvelle base de référence est-elle due ?"""
function new_devex_framework_needed(d::DualSolver, updated_edge_weight::Float64)
    min_abs_devex_iterations = 25
    min_rlv_devex_iterations = 1e-2
    max_allowed_devex_weight_ratio = 3.0
    devex_ratio = max(updated_edge_weight / d.computed_edge_weight,
        d.computed_edge_weight / updated_edge_weight)
    i_te = max(min_abs_devex_iterations,
        trunc(Int, d.solver_num_row / min_rlv_devex_iterations))
    accept_ratio_threshold = max_allowed_devex_weight_ratio^2
    accept_ratio = devex_ratio <= accept_ratio_threshold
    accept_it = d.num_devex_iterations <= i_te
    return !accept_ratio || !accept_it
end

"""`HEkkDual::acceptDualSteepestEdgeWeight` — poids recalculé accepté ou non."""
function accept_dual_steepest_edge_weight!(d::DualSolver,
    updated_edge_weight::Float64)
    accept = updated_edge_weight >=
             kAcceptDseWeightThreshold * d.computed_edge_weight
    assess_dse_weight_error!(d.engine, d.computed_edge_weight,
        updated_edge_weight)
    return accept
end

"""
`HEkk::switchToDevex` : décide la bascule DSE → Devex (coût de la NLA ou
erreur de poids). La bascule elle-même n'est pas portée : `iterate!` lève une
erreur si elle est déclenchée.
"""
function switch_to_devex!(d::DualSolver)
    e = d.engine
    info = e.info
    cost_measure_limit = 1000.0
    cost_min_density = 0.01
    frac_total_before_switch = 0.1
    frac_costly_before_switch = 0.05
    switch_to_devex = false
    denominator = max(max(info.row_ep_density, info.col_aq_density),
        info.row_ap_density)
    if denominator > 0
        info.costly_DSE_measure = (info.row_DSE_density / denominator)^2
    else
        info.costly_DSE_measure = 0.0
    end
    costly_iteration = info.costly_DSE_measure > cost_measure_limit &&
                       info.row_DSE_density > cost_min_density
    info.costly_DSE_frequency =
        (1 - kRunningAverageMultiplier) * info.costly_DSE_frequency
    if costly_iteration
        info.num_costly_DSE_iteration += 1
        info.costly_DSE_frequency += kRunningAverageMultiplier
        local_iteration_count = e.iteration_count - info.control_iteration_count0
        local_num_tot = e.lp.num_col + e.lp.num_row
        switch_to_devex =
            info.allow_dual_steepest_edge_to_devex_switch &&
            info.num_costly_DSE_iteration >
            local_iteration_count * frac_costly_before_switch &&
            local_iteration_count > frac_total_before_switch * local_num_tot
    end
    if !switch_to_devex
        local_measure = info.average_log_low_DSE_weight_error +
                        info.average_log_high_DSE_weight_error
        threshold = e.options.dual_steepest_edge_weight_log_error_threshold
        switch_to_devex = info.allow_dual_steepest_edge_to_devex_switch &&
                          local_measure > threshold
    end
    return switch_to_devex
end

"""`HEkkDual::updatePivots` — base, facteur, vue rowwise, liste."""
function update_pivots!(d::DualSolver)
    d.rebuild_reason != kRebuildReasonNo && return d
    e = d.engine
    transform_for_update!(e.nla, d.col_aq, d.row_ep, d.variable_in, d.row_out)
    update_pivots!(e, d.variable_in, d.row_out, d.move_out)
    e.iteration_count += 1
    d.rebuild_reason = update_factor!(e, d.col_aq, d.row_ep, d.row_out,
        d.rebuild_reason)
    update_matrix!(e, d.variable_in, d.variable_out)
    delete_freelist!(d.dual_row, d.variable_in)
    update_pivots!(d.dual_rhs, d.row_out,
        e.info.workValue[d.variable_in] + d.theta_primal)
    return d
end

"""`HEkkDual::shiftCost` — mémorise un shift de coût (jamais un vrai shift ici)."""
function shift_cost!(d::DualSolver, iCol::Int, amount::Float64)
    info = d.engine.info
    info.costs_shifted = true
    amount == 0.0 && return d
    info.workShift[iCol] = amount
    return d
end

"""`HEkkDual::shiftBack` — annule le shift de coût de la variable sortante."""
function shift_back!(d::DualSolver, iCol::Int)
    info = d.engine.info
    if info.workShift[iCol] != 0.0
        info.workDual[iCol] -= info.workShift[iCol]
        info.workShift[iCol] = 0.0
    end
    return d
end

"""
`HEkkDual::computeDualInfeasibilitiesWithFixedVariableFlips` — compte les
infaisabilités duales selon les bornes de travail (les variables fixes, de
mouvement nul, ne comptent pas).
"""
function compute_dual_infeasibilities_with_fixed_variable_flips!(d::DualSolver)
    e = d.engine
    info = e.info
    tolerance = e.options.dual_feasibility_tolerance
    num = 0
    max_infeasibility = 0.0
    sum_infeasibility = 0.0
    for iVar ∈ 1:d.solver_num_tot
        e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue || continue
        lower = info.workLower[iVar]
        upper = info.workUpper[iVar]
        dual = info.workDual[iVar]
        if lower == -kHighsInf && upper == kHighsInf
            dual_infeasibility = abs(dual)
        else
            dual_infeasibility = -e.basis.nonbasicMove[iVar] * dual
        end
        if dual_infeasibility > 0
            if dual_infeasibility >= tolerance
                num += 1
            end
            max_infeasibility = max(dual_infeasibility, max_infeasibility)
            sum_infeasibility += dual_infeasibility
        end
    end
    info.num_dual_infeasibilities = num
    info.max_dual_infeasibility = max_infeasibility
    info.sum_dual_infeasibilities = sum_infeasibility
    return d
end

"""
`HEkkDual::correctDualInfeasibilities` — supprime les infaisabilités duales :
flip des fixes (et des boxées hors `force_phase2`), shift tiré au hasard sinon.
Rend le nombre d'infaisabilités restantes (variables libres) dans
`d.dual_infeas_count`.
"""
function correct_dual_infeasibilities!(d::DualSolver)
    e = d.engine
    info = e.info
    random = e.random
    tolerance = e.options.dual_feasibility_tolerance
    free_infeasibility_count = 0
    flip_objective_change = 0.0
    shift_objective_change = 0.0
    for iVar ∈ 1:d.solver_num_tot
        e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue || continue
        lower = info.workLower[iVar]
        upper = info.workUpper[iVar]
        current_dual = info.workDual[iVar]
        move = e.basis.nonbasicMove[iVar]
        fixed = lower == upper
        boxed = lower > -kHighsInf && upper < kHighsInf
        free = lower == -kHighsInf && upper == kHighsInf
        if free
            dual_infeasibility = abs(current_dual)
            if dual_infeasibility >= tolerance
                free_infeasibility_count += 1
            end
            continue
        end
        dual_infeasibility = -move * current_dual
        dual_infeasibility < tolerance && continue
        if fixed || (boxed && !d.force_phase2)
            flip_bound!(e, iVar)
            flip = upper - lower
            flip_objective_change += move * flip * current_dual * e.cost_scale
            continue
        end
        # Unilatérale (ou boxée en force_phase2) : shift de coût aléatoire.
        info.allow_cost_shifting ||
            error("DualSolver : shift de coût interdit mais nécessaire")
        info.costs_shifted = true
        if move == kNonbasicMoveUp
            new_dual = (1 + fraction(random)) * tolerance
            shift = new_dual - current_dual
            info.workDual[iVar] = new_dual
            info.workCost[iVar] += shift
        else
            new_dual = -(1 + fraction(random)) * tolerance
            shift = new_dual - current_dual
            info.workDual[iVar] = new_dual
            info.workCost[iVar] += shift
        end
        shift_objective_change += shift * info.workValue[iVar] * e.cost_scale
    end
    d.dual_infeas_count = free_infeasibility_count
    d.force_phase2 = false
    return d
end

"""
`HEkkDual::assessPhase1Optimality` : optimal en phase 1 avec objectif dual non
nul. Sans perturbation de coûts, on conclut directement sur les infaisabilités
duales du LP (phase 2, ou infaisabilité duale).
"""
function assess_phase1_optimality!(d::DualSolver)
    e = d.engine
    info = e.info
    if info.costs_perturbed
        cleanup!(d)
    else
        @assert d.dual_infeas_count == 0
        @assert info.dual_objective_value != 0
    end
    assess_phase1_optimality_unperturbed!(d)
    if d.dual_infeas_count > 0
        # Le retour en phase 1 est déjà en place : les valeurs primales doivent
        # changer, la faisabilité primale est inconnue.
        @assert d.solve_phase == kSolvePhase1
    elseif d.solve_phase == kSolvePhase2
        exit_phase1_reset_duals!(d)
    end
    return d
end

"""`HEkkDual::assessPhase1OptimalityUnperturbed` (coûts non perturbés)."""
function assess_phase1_optimality_unperturbed!(d::DualSolver)
    e = d.engine
    info = e.info
    @assert !info.costs_perturbed
    if d.dual_infeas_count == 0
        if info.dual_objective_value == 0
            d.solve_phase = kSolvePhase2
        else
            num_lp_dual_infeasibilities, _, _ =
                compute_simplex_lp_dual_infeasible(e)
            if num_lp_dual_infeasibilities == 0
                d.solve_phase = kSolvePhase2
            else
                e.model_status = kUnboundedOrInfeasible
                d.solve_phase = kSolvePhaseExit
            end
        end
    end
    return d
end

"""`HEkkDual::exitPhase1ResetDuals` — duals des libres ramenés à zéro."""
function exit_phase1_reset_duals!(d::DualSolver)
    e = d.engine
    info = e.info
    if !info.costs_perturbed
        initialise_cost!(e, kDual, kSolvePhase2; perturb=true)
        compute_dual!(e)
    end
    num_shift = 0
    for iVar ∈ 1:d.solver_num_tot
        e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue || continue
        if iVar <= e.lp.num_col
            lp_lower = e.lp.col_lower[iVar]
            lp_upper = e.lp.col_upper[iVar]
        else
            iRow = iVar - e.lp.num_col
            lp_lower = e.lp.row_lower[iRow]
            lp_upper = e.lp.row_upper[iRow]
        end
        if lp_lower == -kHighsInf && lp_upper == kHighsInf
            shift = -info.workDual[iVar]
            info.workDual[iVar] = 0.0
            info.workCost[iVar] += shift
            num_shift += 1
        end
    end
    num_shift > 0 && (info.costs_shifted = true)
    return d
end

"""
`HEkkDual::assessPossiblyDualUnbounded` — preuve d'infaisabilité primale ; si
elle échoue, la source marque un tabou (backtracking M3c) : ici, erreur.
"""
function assess_possibly_dual_unbounded!(d::DualSolver)
    @assert d.rebuild_reason == kRebuildReasonPossiblyDualUnbounded
    d.solve_phase == kSolvePhase2 || return d
    d.engine.status.has_fresh_rebuild || return d
    if proof_of_primal_infeasibility!(d.engine, d.row_ep, d.move_out, d.row_out)
        d.solve_phase = kSolvePhaseExit
        d.engine.model_status = kInfeasible
    else
        # Preuve non concluante : le changement de base est rendu tabou et la
        # reconstruction reprend sans lui (la source fait de même).
        add_bad_basis_change!(d.engine, d.row_out, d.variable_out,
            d.variable_in, kBadBasisChangeFailedInfeasibilityProof, true)
        d.rebuild_reason = kRebuildReasonNo
    end
    return d
end

"""`HEkkDual::bailoutOnDualObjective` (limite d'objectif non portée)."""
function bailout_on_dual_objective!(d::DualSolver)
    e = d.engine
    e.solve_bailout && return true
    if e.lp.sense == kMinimize && d.solve_phase == kSolvePhase2 &&
       e.info.updated_dual_objective_value > e.options.objective_bound
        error("DualSolver : limite d'objectif non portée (M5)")
    end
    return e.solve_bailout
end
