# Portage partiel de `highs/simplex/SimplexStruct.h` (licence MIT, HiGHS).
#
# Tailles conformes à la source : `basicIndex` a `num_row` entrées, les statuts
# couvrent `num_col + num_row` variables (séquence : colonnes `1:num_col`, lignes
# `num_col+1:num_col+num_row`). `nonbasicFlag`/`nonbasicMove` sont des **valeurs
# encodées** (0/1 et -1/0/1), pas des indices : ne pas les décaler.

"""
    SimplexBasis(num_col, num_row)

Base du simplexe. La source laisse les tableaux non initialisés à `setup` ; on
les met à zéro pour un état de départ déterministe.
"""
mutable struct SimplexBasis
    basicIndex::Vector{Int}
    nonbasicFlag::Vector{Int8}
    nonbasicMove::Vector{Int8}
    hash::UInt64
end

function SimplexBasis(num_col::Int, num_row::Int)
    (num_col >= 0 && num_row >= 0) || throw(ArgumentError("tailles négatives"))
    return SimplexBasis(zeros(Int, num_row), zeros(Int8, num_col + num_row),
        zeros(Int8, num_col + num_row), UInt64(0))
end

"""
    SimplexStatus()

Sous-ensemble de `HighsSimplexStatus` nécessaire à M3b. Les drapeaux sont
posés par les routines du moteur (`set_basis!`, `compute_factor!`,
`update_pivots!`).
"""
mutable struct SimplexStatus
    has_basis::Bool
    has_ar_matrix::Bool
    has_invert::Bool
    has_fresh_invert::Bool
    has_fresh_rebuild::Bool
    has_dual_objective_value::Bool
    has_primal_objective_value::Bool
    has_dual_steepest_edge_weights::Bool
end

SimplexStatus() = SimplexStatus(false, false, false, false, false, false, false,
    false)

"""Réinitialise une base — `SimplexBasis::setup` (HSimplex.cpp:36)."""
function setup!(basis::SimplexBasis, num_col::Int, num_row::Int)
    (num_col >= 0 && num_row >= 0) || throw(ArgumentError("tailles négatives"))
    resize!(basis.basicIndex, num_row)
    fill!(basis.basicIndex, 0)
    resize!(basis.nonbasicFlag, num_col + num_row)
    fill!(basis.nonbasicFlag, 0)
    resize!(basis.nonbasicMove, num_col + num_row)
    fill!(basis.nonbasicMove, 0)
    basis.hash = UInt64(0)
    return basis
end

"""Vide une base — `SimplexBasis::clear` (HSimplex.cpp:26)."""
function clear!(basis::SimplexBasis)
    empty!(basis.basicIndex)
    empty!(basis.nonbasicFlag)
    empty!(basis.nonbasicMove)
    basis.hash = UInt64(0)
    return basis
end

"""Copie indépendante d'une base (`SimplexBasis` est mutable)."""
function copy_basis(basis::SimplexBasis)
    return SimplexBasis(copy(basis.basicIndex), copy(basis.nonbasicFlag),
        copy(basis.nonbasicMove), basis.hash)
end

function copy_basis!(dst::SimplexBasis, src::SimplexBasis)
    resize!(dst.basicIndex, length(src.basicIndex))
    copyto!(dst.basicIndex, src.basicIndex)
    resize!(dst.nonbasicFlag, length(src.nonbasicFlag))
    copyto!(dst.nonbasicFlag, src.nonbasicFlag)
    resize!(dst.nonbasicMove, length(src.nonbasicMove))
    copyto!(dst.nonbasicMove, src.nonbasicMove)
    dst.hash = src.hash
    return dst
end

"""
    BadBasisChange(row_out, variable_out, variable_in, reason, taboo)

Enregistrement de `HighsSimplexBadBasisChangeRecord` : changement de base
interdit (tabou) ou simplement suspect, avec la valeur de rangée sauvegardée
par `applyTabooRowOut`.
"""
mutable struct BadBasisChange
    taboo::Bool
    row_out::Int
    variable_out::Int
    variable_in::Int
    reason::Int
    save_value::Float64
end

BadBasisChange(row_out::Int, variable_out::Int, variable_in::Int, reason::Int,
    taboo::Bool) = BadBasisChange(taboo, row_out, variable_out, variable_in,
    reason, 0.0)

"""
    RayRecord(index, sign)

`HighsRayRecord` réduit au rayon primal : `savePrimalRay` y note la variable
entrante et le signe opposé à son mouvement. Le vecteur du rayon et le rayon
dual (rapport, export) ne sont pas portés.
"""
mutable struct RayRecord
    index::Int
    sign::Int
end

RayRecord() = RayRecord(kNoRayIndex, kNoRaySign)

"""`HighsRayRecord::clear`."""
function clear!(ray::RayRecord)
    ray.index = kNoRayIndex
    ray.sign = kNoRaySign
    return ray
end

"""
    SimplexInfo(num_col, num_row)

Sous-ensemble des vecteurs de travail de `HighsSimplexInfo` nécessaires au port.
Les jalons suivants l'étendront ; rien d'autre n'est anticipé ici.
"""
mutable struct SimplexInfo
    workCost::Vector{Float64}
    workDual::Vector{Float64}
    workShift::Vector{Float64}
    workLower::Vector{Float64}
    workUpper::Vector{Float64}
    workRange::Vector{Float64}
    workValue::Vector{Float64}
    baseLower::Vector{Float64}
    baseUpper::Vector{Float64}
    baseValue::Vector{Float64}
    num_primal_infeasibilities::Int
    max_primal_infeasibility::Float64
    sum_primal_infeasibilities::Float64
    num_dual_infeasibilities::Int
    max_dual_infeasibility::Float64
    sum_dual_infeasibilities::Float64
    # Champs M3a, dans l'ordre de `HighsSimplexInfo`.
    workLowerShift::Vector{Float64}
    workUpperShift::Vector{Float64}
    primal_objective_value::Float64
    dual_objective_value::Float64
    num_basic_logicals::Int
    costs_shifted::Bool
    costs_perturbed::Bool
    bounds_shifted::Bool
    bounds_perturbed::Bool
    col_aq_density::Float64
    row_ep_density::Float64
    row_ap_density::Float64
    primal_col_density::Float64
    dual_col_density::Float64
    col_basic_feasibility_change_density::Float64
    row_basic_feasibility_change_density::Float64
    # Champs M3b (dual) : vecteurs aléatoires, compteurs de phase et options
    # recopiées par `setSimplexOptions`.
    numTotRandomValue::Vector{Float64}
    numTotPermutation::Vector{Int}
    numColPermutation::Vector{Int}
    devex_index::Vector{Int}
    col_BFRT_density::Float64
    row_DSE_density::Float64
    col_steepest_edge_density::Float64
    store_squared_primal_infeasibility::Bool
    dual_phase1_iteration_count::Int
    dual_phase2_iteration_count::Int
    updated_dual_objective_value::Float64
    updated_primal_objective_value::Float64
    primal_phase1_iteration_count::Int
    primal_phase2_iteration_count::Int
    primal_bound_swap::Int
    update_count::Int
    simplex_strategy::Int
    primal_simplex_bound_perturbation_multiplier::Float64
    dual_simplex_cost_perturbation_multiplier::Float64
    primal_simplex_phase1_cost_perturbation_multiplier::Float64
    dual_edge_weight_strategy::Int
    price_strategy::Int
    factor_pivot_threshold::Float64
    update_limit::Int
    allow_cost_shifting::Bool
    allow_cost_perturbation::Bool
    allow_bound_perturbation::Bool
    # Contrôle DSE/Devex (`switchToDevex`).
    allow_dual_steepest_edge_to_devex_switch::Bool
    control_iteration_count0::Int
    costly_DSE_frequency::Float64
    num_costly_DSE_iteration::Int
    costly_DSE_measure::Float64
    average_log_low_DSE_weight_error::Float64
    average_log_high_DSE_weight_error::Float64
    # Robustesse : base de backtracking (`getBacktrackingBasis`) et drapeau de
    # reprise de phase après restauration.
    valid_backtracking_basis::Bool
    backtracking_basis::SimplexBasis
    backtracking_basis_costs_shifted::Bool
    backtracking_basis_costs_perturbed::Bool
    backtracking_basis_bounds_shifted::Bool
    backtracking_basis_bounds_perturbed::Bool
    backtracking_basis_workShift::Vector{Float64}
    backtracking_basis_workLowerShift::Vector{Float64}
    backtracking_basis_workUpperShift::Vector{Float64}
    backtracking_basis_edge_weight::Vector{Float64}
    backtracking::Bool
end

function SimplexInfo(num_col::Int, num_row::Int)
    (num_col >= 0 && num_row >= 0) || throw(ArgumentError("tailles négatives"))
    n = num_col + num_row
    return SimplexInfo(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
        zeros(n), zeros(num_row), zeros(num_row), zeros(num_row),
        -1, kHighsIllegalInfeasibilityMeasure, kHighsIllegalInfeasibilityMeasure,
        -1, kHighsIllegalInfeasibilityMeasure, kHighsIllegalInfeasibilityMeasure,
        zeros(n), zeros(n), 0.0, 0.0, 0,
        false, false, false, false, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        zeros(n), zeros(Int, n), zeros(Int, n), zeros(Int, n), 0.0, 0.0, 0.0,
        true, 0, 0, 0.0, 0.0, 0, 0, 0, 0, kSimplexStrategyDualPlain, 1.0, 1.0,
        1.0,
        kSimplexEdgeWeightStrategyDantzig,
        kSimplexPriceStrategyRowSwitchColSwitch, kDefaultPivotThreshold, 5000,
        true, true, true, false, 0, 0.0, 0, 0.0, 0.0, 0.0,
        false, SimplexBasis(0, 0), false, false, false, false,
        Float64[], Float64[], Float64[], Float64[], false)
end

"""
Redimensionne et remet à zéro ; les compteurs d'infaisabilité repassent à -1.
Les densités reprennent celles d'`initialiseControl` (`dual_col_density = 1` :
les coûts sont supposés non nuls).
"""
function setup!(info::SimplexInfo, num_col::Int, num_row::Int)
    (num_col >= 0 && num_row >= 0) || throw(ArgumentError("tailles négatives"))
    n = num_col + num_row
    for v ∈ (info.workCost, info.workDual, info.workShift, info.workLower,
        info.workUpper, info.workRange, info.workValue, info.workLowerShift,
        info.workUpperShift)
        resize!(v, n)
        fill!(v, 0.0)
    end
    for v ∈ (info.baseLower, info.baseUpper, info.baseValue)
        resize!(v, num_row)
        fill!(v, 0.0)
    end
    info.num_primal_infeasibilities = -1
    info.max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure
    info.sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure
    info.num_dual_infeasibilities = -1
    info.max_dual_infeasibility = kHighsIllegalInfeasibilityMeasure
    info.sum_dual_infeasibilities = kHighsIllegalInfeasibilityMeasure
    info.primal_objective_value = 0.0
    info.dual_objective_value = 0.0
    info.num_basic_logicals = 0
    info.costs_shifted = false
    info.costs_perturbed = false
    info.bounds_shifted = false
    info.bounds_perturbed = false
    info.col_aq_density = 0.0
    info.row_ep_density = 0.0
    info.row_ap_density = 0.0
    info.primal_col_density = 0.0
    info.dual_col_density = 1.0
    info.col_basic_feasibility_change_density = 0.0
    info.row_basic_feasibility_change_density = 0.0
    for v ∈ (info.numTotRandomValue, info.numTotPermutation, info.devex_index)
        resize!(v, n)
        isa(v, Vector{Float64}) ? fill!(v, 0.0) : fill!(v, 0)
    end
    resize!(info.numColPermutation, num_col)
    fill!(info.numColPermutation, 0)
    info.col_BFRT_density = 0.0
    info.row_DSE_density = 0.0
    info.col_steepest_edge_density = 0.0
    info.store_squared_primal_infeasibility = true
    info.dual_phase1_iteration_count = 0
    info.dual_phase2_iteration_count = 0
    info.updated_dual_objective_value = 0.0
    info.updated_primal_objective_value = 0.0
    info.primal_phase1_iteration_count = 0
    info.primal_phase2_iteration_count = 0
    info.primal_bound_swap = 0
    info.update_count = 0
    info.simplex_strategy = kSimplexStrategyDualPlain
    info.primal_simplex_bound_perturbation_multiplier = 1.0
    info.dual_simplex_cost_perturbation_multiplier = 1.0
    info.primal_simplex_phase1_cost_perturbation_multiplier = 1.0
    info.dual_edge_weight_strategy = kSimplexEdgeWeightStrategyDantzig
    info.price_strategy = kSimplexPriceStrategyRowSwitchColSwitch
    info.factor_pivot_threshold = kDefaultPivotThreshold
    info.update_limit = 5000
    info.allow_cost_shifting = true
    info.allow_cost_perturbation = true
    info.allow_bound_perturbation = true
    info.allow_dual_steepest_edge_to_devex_switch = false
    info.control_iteration_count0 = 0
    info.costly_DSE_frequency = 0.0
    info.num_costly_DSE_iteration = 0
    info.costly_DSE_measure = 0.0
    info.average_log_low_DSE_weight_error = 0.0
    info.average_log_high_DSE_weight_error = 0.0
    info.valid_backtracking_basis = false
    clear!(info.backtracking_basis)
    info.backtracking_basis_costs_shifted = false
    info.backtracking_basis_costs_perturbed = false
    info.backtracking_basis_bounds_shifted = false
    info.backtracking_basis_bounds_perturbed = false
    empty!(info.backtracking_basis_workShift)
    empty!(info.backtracking_basis_workLowerShift)
    empty!(info.backtracking_basis_workUpperShift)
    empty!(info.backtracking_basis_edge_weight)
    info.backtracking = false
    return info
end

"""Vide les vecteurs de travail et remet les compteurs d'infaisabilité à -1."""
function clear!(info::SimplexInfo)
    for v ∈ (info.workCost, info.workDual, info.workShift, info.workLower,
        info.workUpper, info.workRange, info.workValue, info.baseLower,
        info.baseUpper, info.baseValue, info.workLowerShift,
        info.workUpperShift, info.numTotRandomValue, info.numTotPermutation,
        info.numColPermutation, info.devex_index)
        empty!(v)
    end
    info.num_primal_infeasibilities = -1
    info.max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure
    info.sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure
    info.num_dual_infeasibilities = -1
    info.max_dual_infeasibility = kHighsIllegalInfeasibilityMeasure
    info.sum_dual_infeasibilities = kHighsIllegalInfeasibilityMeasure
    info.primal_objective_value = 0.0
    info.dual_objective_value = 0.0
    info.num_basic_logicals = 0
    info.costs_shifted = false
    info.costs_perturbed = false
    info.bounds_shifted = false
    info.bounds_perturbed = false
    return info
end
