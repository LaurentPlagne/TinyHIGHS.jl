# Port of `simplex/HEkkPrimal.{h,cpp}` (MIT License, HiGHS) — slices M4a
# (plain phase 2), M4b (phase 1, infeasible/unbounded classification), and M4c
# (Devex/steepest edge weights, primal ray, recovery after backtracking).
#
# Ported: `initialiseSolve`, `solve` (preamble with bound perturbation,
# major loop, `kSolvePhaseUnknown` recovery after backtracking),
# `solvePhase1`, `solvePhase2`, `iterate`, `chuzc`/`chooseColumn`, `useVariableIn`,
# `phase1ChooseRow`, `chooseRow`, `considerBoundSwap`, `assessPivot`,
# `updateVerify`, `update`, `updateDual`, `phase1UpdatePrimal`,
# `basicFeasibilityChange*`, `phase2UpdatePrimal`, `considerInfeasibleValueIn`
# (both phases), `adjustPerturbedEquationOut`, `rebuild`, `cleanup`,
# `correctPrimal`, `getBasicPrimalInfeasibility`, `shiftBound`,
# `getNonbasicFreeColumnSet`, `removeNonbasicFreeColumn`,
# `initialiseDevexFramework`/`updateDevex`,
# `computePrimalSteepestEdgeWeights`/`updatePrimalSteepestEdgeWeights`,
# `updateBtranPSE`, and `savePrimalRay` (ray export omitted, §1).
#
# Not ported: hyper-sparse CHUZC, which the frozen source disables in both
# branches of `rebuild` (`use_hyper_chuzc = false`), objective limit,
# and reporting. The check `debugPrimalSteepestEdgeWeights` is ported for
# its side effects (RNG and FTRAN), without reporting. Missing branches
# throw explicit errors rather than silently degrading.
#
# `solve!` assumes fresh state like the HiGHS source: costs, bounds, values,
# primal, dual, and infeasibilities already computed by caller (`HEkk::solve`
# or `HEkkDual::solve` cleanup), and `INVERT` performed.

"""
    PrimalSolver(engine::SimplexEngine)

Primal revised simplex algorithm solver, equivalent to HiGHS's `HEkkPrimal`.

Implements:
- Primal Phase 1 and Phase 2.
- Column selection (`chooseColumn`) with Dantzig, Devex, and Steepest Edge pricing.
- Row selection (`chooseRow`) and Harris two-pass ratio tests.
- Bound swaps and feasibility restoration.
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

"""`HEkkPrimal::initialiseSolve` — Dantzig, Devex, or steepest edge weights."""
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
        # No dual DSE weights to maintain: source assigns vectors
        # (used around factorization and backtracking).
        fill!(e.dual_edge_weight, 1.0)
        resize!(e.scattered_dual_edge_weight, p.solver_num_tot)
    end
    strategy = e.options.simplex_primal_edge_weight_strategy
    if strategy == kSimplexEdgeWeightStrategyChoose ||
       strategy == kSimplexEdgeWeightStrategyDevex
        # "choose" defaults to Devex, matching source.
        p.edge_weight_mode = kEdgeWeightDevex
        initialise_devex_framework!(p)
    elseif strategy == kSimplexEdgeWeightStrategyDantzig
        p.edge_weight_mode = kEdgeWeightDantzig
        fill!(p.edge_weight, 1.0)
    elseif strategy == kSimplexEdgeWeightStrategySteepestEdge
        p.edge_weight_mode = kEdgeWeightSteepestEdge
        compute_primal_steepest_edge_weights!(p)
    else
        error("PrimalSolver: unknown edge weight strategy $strategy")
    end
    return p
end

"""
`HEkkPrimal::initialiseDevexFramework`: unit weights, reference framework
`devex_index = nonbasicFlag²` (nonbasics at initialization time),
counters reset to zero.
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

"""`HEkkPrimal::updateDevex` — Devex weights after pivot."""
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
    # Outlying weight: counted, source reinitializes above threshold.
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
`HEkkPrimal::computePrimalSteepestEdgeWeights`: exact weights. On logical
basis, `1 + ‖a_j‖²` (sum in column order); otherwise one FTRAN per
nonbasic variable.
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

"""`HEkkPrimal::computePrimalSteepestEdgeWeight` — column FTRAN."""
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
`HEkkPrimal::updatePrimalSteepestEdgeWeights`: updates weights using
vector `mu = B^{-T} hat{a}_q` (BTRAN PSE), on nonbasics of pivot row
(structurals then slacks).
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
    # Outgoing tableau column is pivot column divided by pivot,
    # except at pivot position (1/pivot).
    p.edge_weight[p.variable_out] =
        (1 + col_aq_squared_2norm) / (p.alpha_col * p.alpha_col)
    p.edge_weight[p.variable_in] = 0.0
    return p
end

"""
`HEkkPrimal::debugPrimalSteepestEdgeWeights` — expensive branch triggered by
`update` in steepest edge strategy. Source only prints on error,
but check consumes RNG (`integer(num_tot)` until nonbasic) and runs
one FTRAN per drawn variable, updating `col_aq_density`: its side effects
are ported, without reporting.
"""
function check_primal_steepest_edge_weights!(p::PrimalSolver)
    e = p.engine
    num_check_weight = max(1, min(10, p.solver_num_tot ÷ 10))
    local_col_aq = HVector(p.solver_num_row)
    for _ ∈ 1:num_check_weight
        iVar = 0
        while true
            # `HighsRandom::integer` is 0-based in source.
            iVar = integer(e.random, p.solver_num_tot) + 1
            e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue && break
        end
        compute_primal_steepest_edge_weight(p, iVar, local_col_aq)
    end
    return p
end

"""`HEkkPrimal::updateBtranPSE` — BTRAN of weight vector (dedicated density)."""
function update_btran_pse!(p::PrimalSolver)
    e = p.engine
    btran!(e.nla, p.col_steepest_edge, e.info.col_steepest_edge_density)
    e.info.col_steepest_edge_density = update_operation_result_density(
        e.info.col_steepest_edge_density,
        p.col_steepest_edge.count * p.inv_solver_num_row)
    return p
end

"""`HEkkPrimal::getNonbasicFreeColumnSet` — ascending column order."""
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

`HEkkPrimal::solve` (M4a). `force_phase2` is `pass_force_phase2` in
source: dual cleanup calls it with `true`. Returns the engine; status is
in `engine.model_status`. Dual cleanup following `OptimalCleanup` is
wired; primal ray (unboundedness) and phase 1 are supported.
"""
function solve!(p::PrimalSolver; force_phase2::Bool=false, restore::Bool=true)
    e = p.engine
    # `Highs::run` rejected bounds: LP is infeasible without simplex.
    e.bounds_infeasible && return e
    initialise_solve!(p)
    e.status.has_invert || error("PrimalSolver: INVERT required before solve")
    # `HEkk::solve`: lift blocks inherited from previous solve and
    # clear ray from earlier solve.
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
        # Shifted bounds: values and infeasibilities are recomputed on
        # perturbed bounds.
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
            # Recovery after backtracking: primal infeasibility count
            # yields phase, and costs/values of restored basis are reset.
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
        # `solvePhase1` that cleaned up bounds stays in phase 1: major loop
        # continues. `solvePhase2` can return to phase 1 after cleanup.
        # Source exits on optimal, exit, and dual cleanup.
        (p.solve_phase == kSolvePhaseOptimal ||
         p.solve_phase == kSolvePhaseExit ||
         p.solve_phase == kSolvePhaseOptimalCleanup) && break
    end
    p.solve_phase == kSolvePhaseOptimal && (e.model_status = kOptimal)
    if p.solve_phase == kSolvePhaseOptimalCleanup
        # Primal infeasibilities after phase 2: dual feasible, so dual
        # cleans up (without cost perturbation, plain dual strategy).
        compute_primal_objective_value!(e)
        save_cost_perturbation =
            e.info.dual_simplex_cost_perturbation_multiplier
        e.info.dual_simplex_cost_perturbation_multiplier = 0.0
        save_strategy = e.info.simplex_strategy
        e.info.simplex_strategy = kSimplexStrategyDualPlain
        d = DualSolver(e)
        # The `kUnboundedOrInfeasible` classification belongs to `HEkk::solve`,
        # not the nested cleanup call; scales as well.
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

`HEkkPrimal::solvePhase1`: rebuild/iteration loop in phase 1. Yields on
phase 2 (no more infeasibilities), on proven infeasibility (`kInfeasible`),
or after a `cleanup` that leaves phase 1 to resume.
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
        # `rebuild!` found a primally feasible basis: return to phase 2.
        p.solve_phase == kSolvePhase2 && break
        while true
            iterate!(p)
            bailout!(e) && return p
            p.solve_phase == kSolvePhaseError && return p
            p.solve_phase == kSolvePhase1 || error(
                "PrimalSolver: phase $(p.solve_phase) after iteration (phase 1)")
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
        # Optimal in phase 1: either shifted bounds hid feasibility
        # (cleanup and retry), or infeasibility is proven.
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
        # rebuild!: singularity, or return to phase 1, or exit.
        rebuild!(p)
        p.solve_phase == kSolvePhaseError && return p
        # Backtracking: major loop resumes control and resets phase
        # (`kSolvePhaseUnknown`).
        p.solve_phase == kSolvePhaseUnknown && return p
        bailout!(e) && return p
        # `rebuild!` found a primal infeasibility: return to phase 1,
        # major loop resumes.
        p.solve_phase == kSolvePhase1 && break
        while true
            iterate!(p)
            bailout!(e) && return p
            p.solve_phase == kSolvePhaseError && return p
            p.solve_phase == kSolvePhase2 || error(
                "PrimalSolver: phase $(p.solve_phase) after iteration")
            p.rebuild_reason != kRebuildReasonNo && break
        end
        # Fresh rebuild data and no flips: evaluate outcome before looping.
        finished = e.status.has_fresh_rebuild && p.num_flip_since_rebuild == 0 &&
                   !rebuild_refactor(e, p.rebuild_reason)
        if finished && taboo_bad_basis_change(e)
            # Only possible basis change is taboo: cannot conclude.
            p.solve_phase = kSolvePhaseTabooBasis
            return p
        end
        finished && break
    end
    p.solve_phase == kSolvePhase1 && return p
    if p.variable_in == -1
        # No CHUZC candidate even after rebuild: likely optimal.
        cleanup!(p)
        if e.info.num_primal_infeasibilities > 0
            p.solve_phase = kSolvePhaseOptimalCleanup
        else
            p.solve_phase = kSolvePhaseOptimal
            e.model_status = kOptimal
            compute_dual_objective_value!(e)
        end
    elseif p.row_out == kNoRowSought
        # CHUZR did not occur (recomputed reduced cost unpromising,
        # without rebuild): rare case unhandled by source as well.
        error("PrimalSolver: row_out = kNoRowSought (rare case not ported)")
    else
        # No CHUZR candidate: primal unbounded, or return to phase 1 after cleanup.
        if e.info.bounds_shifted || e.info.bounds_perturbed
            cleanup!(p)
            if e.info.num_primal_infeasibilities > 0
                p.solve_phase = kSolvePhase1
            end
        else
            # Certified unbounded: primal ray is recorded (incoming
            # variable and sign), then status is set.
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
            # No pivot candidate in phase 1: error, matching source.
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
        # No more infeasibilities in phase 1: force rebuild to switch to phase 2.
        p.rebuild_reason = kRebuildReasonPossiblyPhase1Feasible
    end
    ok_rebuild_reason =
        p.rebuild_reason == kRebuildReasonNo ||
        p.rebuild_reason == kRebuildReasonPossiblyPhase1Feasible ||
        p.rebuild_reason == kRebuildReasonPrimalInfeasibleInPrimalSimplex ||
        p.rebuild_reason == kRebuildReasonSyntheticClockSaysInvert ||
        p.rebuild_reason == kRebuildReasonUpdateLimitReached
    ok_rebuild_reason ||
        error("PrimalSolver: unexpected rebuild_reason $(p.rebuild_reason)")
    return p
end

"""`HEkkPrimal::chuzc` — mask taboos, choose incoming column."""
function chuzc!(p::PrimalSolver)
    e = p.engine
    work_dual = e.info.workDual
    apply_taboo_variable_in!(e, work_dual, 0.0)
    choose_column!(p)
    unapply_taboo_variable_in!(e, work_dual)
    return p
end

"""`HEkkPrimal::chooseColumn` (Dantzig, non-hyper-sparse)."""
function choose_column!(p::PrimalSolver)
    e = p.engine
    work_dual = e.info.workDual
    best_measure = 0.0
    p.variable_in = -1
    # Nonbasic free columns first.
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

"""`HEkk::pivotColumnFtran` — pivot column via FTRAN."""
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

"""`HEkk::computeDualForTableauColumn` — recomputed dual of incoming column."""
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

"""`HEkkPrimal::useVariableIn` — FTRAN then check recomputed dual."""
function use_variable_in!(p::PrimalSolver)
    e = p.engine
    info = e.info
    updated_theta_dual = info.workDual[p.variable_in]
    p.move_in = updated_theta_dual > 0 ? -1 : 1
    move = e.basis.nonbasicMove[p.variable_in]
    if move != 0 && move != p.move_in
        error("PrimalSolver: move_in incompatible with nonbasicMove")
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

"""`HEkkPrimal::chooseRow` — ratio test (passes 1 and 2)."""
function choose_row!(p::PrimalSolver)
    e = p.engine
    info = e.info
    p.row_out = kNoRowChosen
    alpha_tol = info.update_count < 10 ? 1e-9 :
                info.update_count < 20 ? 1e-8 : 1e-7
    # Pass 1: smallest relaxed theta (infeasibility tolerance).
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
    # Pass 2: largest |alpha| among rows at relaxed theta.
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

`HEkkPrimal::phase1ChooseRow`: phase 1 ratio test. Candidates are thetas
where the slope of primal infeasibility changes (`ph1_sorter_r`, theta
relaxed by tolerance) and those where an infeasibility is resolved or created
(`ph1_sorter_t`, tight theta). Selected theta is the last one before slope
becomes negative; pivot is largest `|alpha|` among tight candidates below this
theta, within 10% of maximum.

The signed marker matches C++ `(theta, iRow)`, `(theta, iRow - m)`:
positive = upper bound candidate, negative = lower bound, encoded as
`iRow - m - 1` in 1-based indexing so sort order and ties match the source.
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
            # Basic variable decreases.
            if info.baseValue[iRow] >
               info.baseUpper[iRow] + p.primal_feasibility_tolerance
                # Becomes feasible on reaching its upper bound.
                feas_theta = (info.baseValue[iRow] - info.baseUpper[iRow] -
                              p.primal_feasibility_tolerance) / alpha
                push!(p.ph1_sorter_r, (feas_theta, iRow))
                push!(p.ph1_sorter_t, (feas_theta, iRow))
            end
            if info.baseValue[iRow] >
               info.baseLower[iRow] - p.primal_feasibility_tolerance &&
               info.baseLower[iRow] > -kHighsInf
                # Becomes infeasible again on falling below its lower bound.
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
            # Basic variable increases.
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
    # `i_last` is 1-based index of first theta that is too large;
    # excluded from backward pass, matching 0-based `iLast` in source.
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

"""`HEkkPrimal::considerBoundSwap` — primal theta, flip, or pivot."""
function consider_bound_swap!(p::PrimalSolver)
    e = p.engine
    info = e.info
    if p.row_out == kNoRowChosen
        # No blocking row: flip or unbounded.
        p.theta_primal = p.move_in * kHighsInf
        p.move_out = 0
    else
        p.alpha_col = p.col_aq.array[p.row_out]
        # In phase 1, `move_out` comes from `phase1ChooseRow`: outgoing variable
        # may become feasible (toward its bound) or remain so; direction is not
        # deducible from pivot sign.
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
        # Possible unboundedness: in phase 1, row_out >= 0 is guaranteed
        # (its absence is treated as an error in `iterate!`).
        p.rebuild_reason = kRebuildReasonPossiblyPrimalUnbounded
    end
    return p
end

"""`HEkk::unitBtran` — BTRAN of unit vector e_row_out."""
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

"""`HEkkPrimal::assessPivot` — unit BTRAN, PRICE, numerical checks."""
function assess_pivot!(p::PrimalSolver)
    e = p.engine
    p.alpha_col = p.col_aq.array[p.row_out]
    p.variable_out = e.basis.basicIndex[p.row_out]
    unit_btran!(p, p.row_out, p.row_ep)
    tableau_row_price!(e, p.row_ep, p.row_ap)
    update_verify!(p)
    return p
end

"""`HEkkPrimal::updateVerify` — column pivot vs row pivot."""
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

"""`HEkkPrimal::adjustPerturbedEquationOut` — fixed variable leaves basis."""
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
    # Outgoing variable is fixed: primal theta rectified to its true value.
    true_fixed_value = lp_lower
    p.theta_primal =
        (info.baseValue[p.row_out] - true_fixed_value) / p.alpha_col
    info.workLower[p.variable_out] = true_fixed_value
    info.workUpper[p.variable_out] = true_fixed_value
    info.workRange[p.variable_out] = 0.0
    p.value_in = info.workValue[p.variable_in] + p.theta_primal
    return p
end

"""`HEkkPrimal::update` — primal, dual, factor, and pivot updates."""
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
            error("PrimalSolver: inconsistent flip")
        e.basis.nonbasicMove[p.variable_in] = -p.move_in
    else
        adjust_perturbed_equation_out!(p)
    end
    # Matching source order: primal update, then dual. In phase 1,
    # duals are adjusted for feasibility changes (hyper-sparse CHUZC,
    # not ported, inserts here).
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
        # The expensive check in source (before update) is ported
        # for its side effects: RNG draws and verification FTRAN.
        check_primal_steepest_edge_weights!(p)
        update_primal_steepest_edge_weights!(p)
    end
    remove_nonbasic_free_column!(p)
    if e.status.has_dual_steepest_edge_weights
        # Primal must maintain dual DSE weights: dual can resume
        # control (during `OptimalCleanup`) and use them.
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
    # Too many outlying Devex weights: reinitialize reference framework.
    p.edge_weight_mode == kEdgeWeightDevex &&
        p.num_bad_devex_weight > kAllowedNumBadDevexWeight &&
        initialise_devex_framework!(p)
    e.total_synthetic_tick += p.col_aq.synthetic_tick
    e.total_synthetic_tick += p.row_ep.synthetic_tick
    return p
end

"""`HEkkPrimal::updateDual` — duals after primal simplex step."""
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

`HEkkPrimal::phase1ComputeDual`: phase 1 costs and duals. `workCost` is
`±1` on infeasible basic variables (scaled by phase 1 perturbation
multiplier, set to 1), then `workDual = -[A I]' B^{-T} workCost` restricted
to nonbasics — a full BTRAN and full PRICE.
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
        error("PrimalSolver: phase 1 without infeasibility (zero RHS)")
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
`HEkkPrimal::phase1UpdatePrimal` — base values and phase 1 costs; basic cost
changes are collected in `col_basic_feasibility_change` for dual correction.
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
            # Slack costs have no component in PRICE: their delta is
            # applied directly to the dual.
            iCol > p.solver_num_col && (info.workDual[iCol] += delta_cost)
        end
    end
    invalidate_primal_max_sum_infeasibility_record!(e)
    return p
end

"""
`HEkkPrimal::basicFeasibilityChangeUpdateDual`: BTRAN of basic cost change
vector, PRICE on nonbasics, then component subtraction — slacks receive
the BTRAN vector (their column is the identity).
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

"""`HEkkPrimal::basicFeasibilityChangeBtran` — BTRAN of change vector."""
function basic_feasibility_change_btran!(p::PrimalSolver)
    e = p.engine
    btran!(e.nla, p.col_basic_feasibility_change,
        e.info.col_basic_feasibility_change_density)
    e.info.col_basic_feasibility_change_density = update_operation_result_density(
        e.info.col_basic_feasibility_change_density,
        p.col_basic_feasibility_change.count * p.inv_solver_num_row)
    return p
end

"""`HEkkPrimal::basicFeasibilityChangePrice` — PRICE of change vector."""
function basic_feasibility_change_price!(p::PrimalSolver)
    e = p.engine
    local_density = p.col_basic_feasibility_change.count / p.solver_num_row
    use_col_price, use_row_price_w_switch = choose_price_technique(e,
        local_density)
    clear!(p.row_basic_feasibility_change)
    if use_col_price
        price_by_column!(e.lp.a_matrix, p.row_basic_feasibility_change,
            p.col_basic_feasibility_change)
        # Column PRICE also covers basic variables: zero them out via
        # nonbasic mask.
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
`HEkkPrimal::phase2UpdatePrimal` — basic values, infeasibilities handled by
bound shifting (HiGHS `Always` strategy), objective updated.
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
        # Always strategy: violation is absorbed by bound shift.
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

"""`HEkkPrimal::considerInfeasibleValueIn` — infeasible incoming value."""
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
        # Phase 1: infeasible incoming value becomes a basic cost
        # (perturbed by phase 1 multiplier) and shifts the dual.
        info.num_primal_infeasibilities += 1
        base = info.primal_simplex_phase1_cost_perturbation_multiplier * 5e-7
        cost = Float64(bound_violated)
        base != 0.0 && (cost *= 1 + base * info.numTotRandomValue[p.row_out])
        info.workCost[p.variable_in] = cost
        info.workDual[p.variable_in] += cost
        invalidate_primal_max_sum_infeasibility_record!(e)
        return p
    end
    # Phase 2, Always strategy: bound shift.
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

"""`HEkkPrimal::removeNonbasicFreeColumn` — incoming column becomes basic."""
function remove_nonbasic_free_column!(p::PrimalSolver)
    e = p.engine
    e.basis.nonbasicMove[p.variable_in] == 0 || return p
    idx = findfirst(==(p.variable_in), p.nonbasic_free_col_set)
    idx === nothing &&
        error("PrimalSolver: nonbasic free column absent from set")
    deleteat!(p.nonbasic_free_col_set, idx)
    return p
end

"""
`HEkkPrimal::updateDualSteepestEdgeWeights` — maintain dual DSE weights
during primal phase (dual can resume control during cleanup).
"""
function update_dual_steepest_edge_weights!(p::PrimalSolver)
    e = p.engine
    copy!(p.col_steepest_edge, p.row_ep)
    # `updateFtranDSE`: unapply row scale then FTRAN in scaled space.
    unapply_basis_matrix_row_scale!(e.nla, p.col_steepest_edge)
    ftran_in_scaled_space!(e.nla, p.col_steepest_edge, e.info.row_DSE_density)
    e.info.row_DSE_density = update_operation_result_density(
        e.info.row_DSE_density, p.col_steepest_edge.count * p.inv_solver_num_row)
    edge_weight = e.dual_edge_weight
    # simplex_in_scaled_space_: norm of row_ep directly.
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

"""`HEkkPrimal::shiftBound` — bound and shift to make value feasible."""
function shift_bound!(p::PrimalSolver, lower::Bool, iVar::Int,
    value::Float64, random_value::Float64)
    feasibility = (1 + random_value) * p.primal_feasibility_tolerance
    if lower
        value < p.engine.info.workLower[iVar] - p.primal_feasibility_tolerance ||
            error("PrimalSolver: shiftBound lower without violation")
        infeasibility = p.engine.info.workLower[iVar] - value
        shift = infeasibility + feasibility
        bound = p.engine.info.workLower[iVar] - shift
    else
        value > p.engine.info.workUpper[iVar] + p.primal_feasibility_tolerance ||
            error("PrimalSolver: shiftBound upper without violation")
        infeasibility = value - p.engine.info.workUpper[iVar]
        shift = infeasibility + feasibility
        bound = p.engine.info.workUpper[iVar] + shift
    end
    return bound, shift
end

"""
`HEkkPrimal::savePrimalRay`: incoming variable and sign opposite to its movement.
Ray vector export omitted (§1).
"""
function save_primal_ray!(p::PrimalSolver)
    p.variable_in >= 0 ||
        error("PrimalSolver: primal ray without incoming variable")
    p.move_in != kNoRaySign || error("PrimalSolver: primal ray without sign")
    ray = p.engine.primal_ray_record
    clear!(ray)
    ray.index = p.variable_in
    ray.sign = -p.move_in
    return p
end

"""`HEkkPrimal::getBasicPrimalInfeasibility` — num/max/sum over basics."""
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

"""`HEkkPrimal::rebuild` — INVERT, primal, phase, duals, and objective."""
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
        # Only occurs during backtracking.
        info.backtracking || error("PrimalSolver: ar_matrix absent")
        initialise_partitioned_rowwise_matrix!(e)
    end
    if info.backtracking
        # Backtracking can change phase: exit.
        p.solve_phase = kSolvePhaseUnknown
        return p
    end
    compute_primal!(e)
    p.solve_phase == kSolvePhase2 && correct_primal!(p)
    get_basic_primal_infeasibility!(p)
    if info.num_primal_infeasibilities > 0
        # Primal infeasibilities: source switches to phase 1 and recomputes
        # phase 1 costs and duals.
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

"""`HEkkPrimal::correctPrimal` — shifts on rebuild infeasibilities."""
function correct_primal!(p::PrimalSolver, initialise::Bool=false)
    e = p.engine
    info = e.info
    if initialise
        p.max_max_primal_correction = 0.0
        return true
    end
    p.solve_phase == kSolvePhase2 ||
        error("PrimalSolver: correctPrimal outside phase 2")
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

"""`HEkkPrimal::cleanup` — remove bound perturbations and shifts."""
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
