# Portage partiel de `simplex/HEkk.{h,cpp}` (licence MIT, HiGHS) — tranche M3a
# « état et valeurs ». Voir docs/architecture/portage-julia-simplexe-highs.md §3.
#
# Porté : `setBasis` (base logique), `setNonbasicMove`, `initialiseLpColBound`/
# `initialiseLpRowBound`/`initialiseBound`, `initialiseLpColCost`/
# `initialiseLpRowCost`/`initialiseCost`, `initialiseNonbasicValueAndMove`,
# `computePrimal`, `computeDual`, `fullBtran`/`fullPrice`, objectifs primal et
# dual.
#
# Non porté (M3b/M3c) : itérations duales/primales, choix de rangée et de
# colonne, DSE/Devex, perturbations et shifts, ré-inversions. La perturbation
# aléatoire (bornes primales, coûts duaux) exige `numTotRandomValue_` (RNG
# HiGHS) : la branche lève une erreur explicite plutôt que d'être ignorée.
#
# Le système résolu est celui de HiGHS : chaque ligne porte une variable
# logique `s = -a_i x`, de bornes `[-row_upper, -row_lower]`, si bien que
# `[A I] [x; s] = 0` — le second membre du LP est porté par les bornes des
# logiques, pas par un vecteur b.

"""
    SimplexLp(num_col, num_row, a_matrix, col_cost, col_lower, col_upper,
              row_lower, row_upper; offset = 0.0, sense = kMinimize)

Linear programming model definition.

# Fields
- `num_col::Int`: Number of structural columns (variables).
- `num_row::Int`: Number of constraints (rows).
- `a_matrix::SparseMatrix`: Constraint matrix in 1-based column-wise (CSC) representation.
- `col_cost::Vector{Float64}`: Objective linear coefficients (length `num_col`).
- `col_lower::Vector{Float64}`: Variable lower bounds (length `num_col`).
- `col_upper::Vector{Float64}`: Variable upper bounds (length `num_col`).
- `row_lower::Vector{Float64}`: Constraint lower bounds (length `num_row`).
- `row_upper::Vector{Float64}`: Constraint upper bounds (length `num_row`).
- `offset::Float64`: Objective constant offset (defaults to `0.0`).
- `sense::ObjSense`: Optimization direction (`kMinimize` or `kMaximize`).
- `scale::Scale`: Optional scaling factors.
- `is_scaled::Bool`: Whether the model is currently scaled.
"""
mutable struct SimplexLp
    num_col::Int
    num_row::Int
    a_matrix::SparseMatrix
    col_cost::Vector{Float64}
    col_lower::Vector{Float64}
    col_upper::Vector{Float64}
    row_lower::Vector{Float64}
    row_upper::Vector{Float64}
    offset::Float64
    sense::ObjSense
    scale::Scale
    is_scaled::Bool
end

function SimplexLp(num_col::Int, num_row::Int, a_matrix::SparseMatrix,
    col_cost::Vector{Float64}, col_lower::Vector{Float64},
    col_upper::Vector{Float64}, row_lower::Vector{Float64},
    row_upper::Vector{Float64}; offset::Float64=0.0,
    sense::ObjSense=kMinimize)
    (num_col >= 0 && num_row >= 0) || throw(ArgumentError("dimensions must be non-negative"))
    (a_matrix.num_col == num_col && a_matrix.num_row == num_row) ||
        throw(ArgumentError("dimensions of a_matrix mismatch num_col/num_row"))
    is_colwise(a_matrix) ||
        throw(ArgumentError("a_matrix must be column-wise"))
    (length(col_cost) == num_col && length(col_lower) == num_col &&
     length(col_upper) == num_col) ||
        throw(ArgumentError("column vector lengths mismatch num_col"))
    (length(row_lower) == num_row && length(row_upper) == num_row) ||
        throw(ArgumentError("row vector lengths mismatch num_row"))
    return SimplexLp(num_col, num_row, a_matrix, col_cost, col_lower, col_upper,
        row_lower, row_upper, offset, sense, Scale(), false)
end

"""
    SimplexOptions(; ...)

Configuration parameters for the simplex engine, matching HiGHS default settings.

Key options:
- `primal_feasibility_tolerance`: Primal tolerance (default `1e-7`).
- `dual_feasibility_tolerance`: Dual tolerance (default `1e-7`).
- `time_limit`: Time limit in seconds (default `Inf`).
- `simplex_iteration_limit`: Maximum number of simplex iterations.
- `simplex_dual_edge_weight_strategy`: Dual pricing edge weight strategy (`kSimplexEdgeWeightStrategyChoose`, `kSimplexEdgeWeightStrategyDantzig`, `kSimplexEdgeWeightStrategyDevex`, `kSimplexEdgeWeightStrategySteepestEdge`).
- `simplex_update_limit`: Maximum number of basis updates before full refactorization (default `5000`).
- `random_seed`: Seed for deterministic pseudo-random number generator (perturbations).
"""
struct SimplexOptions
    cost_scale_factor::Int
    dual_simplex_cost_perturbation_multiplier::Float64
    primal_simplex_bound_perturbation_multiplier::Float64
    primal_feasibility_tolerance::Float64
    dual_feasibility_tolerance::Float64
    small_matrix_value::Float64
    time_limit::Float64
    simplex_iteration_limit::Int
    objective_bound::Float64
    objective_target::Float64
    simplex_dual_edge_weight_strategy::Int
    simplex_primal_edge_weight_strategy::Int
    simplex_scale_strategy::Int
    allowed_matrix_scale_factor::Int
    simplex_price_strategy::Int
    simplex_update_limit::Int
    max_dual_simplex_cleanup_level::Int
    max_dual_simplex_phase1_cleanup_level::Int
    no_unnecessary_rebuild_refactor::Bool
    rebuild_refactor_solution_error_tolerance::Float64
    dual_simplex_pivot_growth_tolerance::Float64
    dual_steepest_edge_weight_log_error_threshold::Float64
    random_seed::Int
end

SimplexOptions(; cost_scale_factor::Int=0,
    dual_simplex_cost_perturbation_multiplier::Float64=1.0,
    primal_simplex_bound_perturbation_multiplier::Float64=1.0,
    primal_feasibility_tolerance::Float64=1e-7,
    dual_feasibility_tolerance::Float64=1e-7,
    small_matrix_value::Float64=1e-9,
    time_limit::Float64=kHighsInf,
    simplex_iteration_limit::Int=kHighsIInf,
    objective_bound::Float64=kHighsInf,
    objective_target::Float64=kHighsInf,
    simplex_dual_edge_weight_strategy::Int=kSimplexEdgeWeightStrategyChoose,
    simplex_primal_edge_weight_strategy::Int=kSimplexEdgeWeightStrategyChoose,
    simplex_scale_strategy::Int=kSimplexScaleStrategyOff,
    allowed_matrix_scale_factor::Int=kDefaultAllowedMatrixPow2Scale,
    simplex_price_strategy::Int=kSimplexPriceStrategyRowSwitchColSwitch,
    simplex_update_limit::Int=5000,
    max_dual_simplex_cleanup_level::Int=1,
    max_dual_simplex_phase1_cleanup_level::Int=2,
    no_unnecessary_rebuild_refactor::Bool=true,
    rebuild_refactor_solution_error_tolerance::Float64=1e-8,
    dual_simplex_pivot_growth_tolerance::Float64=1e-9,
    dual_steepest_edge_weight_log_error_threshold::Float64=10.0,
    random_seed::Int=0) =
    SimplexOptions(cost_scale_factor, dual_simplex_cost_perturbation_multiplier,
        primal_simplex_bound_perturbation_multiplier,
        primal_feasibility_tolerance, dual_feasibility_tolerance,
        small_matrix_value, time_limit, simplex_iteration_limit,
        objective_bound, objective_target, simplex_dual_edge_weight_strategy,
        simplex_primal_edge_weight_strategy, simplex_scale_strategy,
        allowed_matrix_scale_factor, simplex_price_strategy,
        simplex_update_limit,
        max_dual_simplex_cleanup_level, max_dual_simplex_phase1_cleanup_level,
        no_unnecessary_rebuild_refactor,
        rebuild_refactor_solution_error_tolerance,
        dual_simplex_pivot_growth_tolerance,
        dual_steepest_edge_weight_log_error_threshold, random_seed)

"""
    SimplexEngine(lp, options = SimplexOptions(); basis = nothing)

Core revised simplex solver engine equivalent to HiGHS's `HEkk`.

Encapsulates:
- The linear model (`lp::SimplexLp`).
- Solver configuration options (`options::SimplexOptions`).
- Current basis state (`basis::SimplexBasis`).
- Sized persistent working buffers (`info::SimplexInfo`), reused across solves without reallocating.
- Numerical Linear Algebra and basis LU factorization state (`nla::Nla`).
- Pseudo-random number generator for numerical perturbation (`random::HighsRandom`).

# Warm-Starting
`SimplexEngine` is designed for high-frequency repeated solves: modify bounds or costs using
`change_col_bounds!`, `change_row_bounds!`, `change_cols_cost!` and call `solve!(engine)` to
perform warm-start re-optimization with zero memory allocation.
"""
mutable struct SimplexEngine
    lp::SimplexLp
    options::SimplexOptions
    basis::SimplexBasis
    info::SimplexInfo
    nla::Nla
    cost_scale::Float64
    status::SimplexStatus
    random::HighsRandom
    iteration_count::Int
    # Compteur du solve courant : `HApp::solveLpSimplex` recopie
    # `highs_info.simplex_iteration_count` (remis à zéro par `Highs::run`)
    # dans `ekk_instance.iteration_count_` à chaque résolution, ce qui rend
    # `simplex_iteration_limit` **par solve** alors que `iteration_count` du
    # port est cumulé (comme `highs_info`, qui sert au rapport).
    iteration_count0::Int
    total_synthetic_tick::Float64
    build_synthetic_tick::Float64
    dual_edge_weight::Vector{Float64}
    scattered_dual_edge_weight::Vector{Float64}
    ar_matrix::SparseMatrix
    model_status::ModelStatus
    solve_bailout::Bool
    solve_start_time::Float64
    dual_simplex_cleanup_level::Int
    dual_simplex_phase1_cleanup_level::Int
    visited_basis::Set{UInt64}
    previous_iteration_cycling_detected::Int
    bad_basis_change::Vector{BadBasisChange}
    primal_ray_record::RayRecord
    # `Highs::run` : bornes significativement incohérentes, le LP est déclaré
    # infaisable sans toucher au simplexe (état posé par `initialise_for_solve!`).
    bounds_infeasible::Bool
    # `HighsBasis basis_` du niveau Highs : base rendue par le dernier solve,
    # rafraîchie à chaque fin de résolution. Elle sert de warm start quand la
    # base du Ekk a été jetée sans que le modèle change de base (changement de
    # coefficient : `HEkk::clear` puis `HEkk::setBasis(basis_)`).
    highs_basis::SimplexBasis
    highs_basis_valid::Bool
    primal_col::HVector
    dual_col::HVector
    dual_row::HVector
    basic_index_before::Vector{Int}
    fse_solution_value::Vector{Float64}
    fse_solution_index::Vector{Int}
    fse_solution_nonzero::Vector{Bool}
    fse_btran_scattered::Vector{Float64}
    fse_random::HighsRandom
end

function SimplexEngine(lp::SimplexLp, options::SimplexOptions=SimplexOptions();
    basis::Union{Nothing,SimplexBasis}=nothing)
    provided_basis = basis !== nothing
    if !provided_basis
        basis = SimplexBasis(lp.num_col, lp.num_row)
    else
        (length(basis.basicIndex) == lp.num_row &&
         length(basis.nonbasicFlag) == lp.num_col + lp.num_row &&
         length(basis.nonbasicMove) == lp.num_col + lp.num_row) ||
            throw(ArgumentError("base incompatible avec les dimensions du LP"))
    end
    info = SimplexInfo(lp.num_col, lp.num_row)
    factor = HFactor(lp.num_col, lp.num_row, lp.num_row, lp.a_matrix.start,
        lp.a_matrix.index, lp.a_matrix.value, basis.basicIndex)
    basis.basicIndex = factor.basic_index
    status = SimplexStatus()
    # Une base fournie est une base : `initialise_for_solve!` ne doit pas la
    # remplacer par la base logique. Son hachage se déduit de `nonbasicFlag`.
    status.has_basis = provided_basis
    provided_basis && (basis.hash = basis_hash(basis))
    engine = SimplexEngine(lp, options, basis, info,
        Nla(factor, lp.num_col, lp.num_row), 1.0, status,
        HighsRandom(options.random_seed), 0, 0, 0.0, 0.0,
        ones(lp.num_row), zeros(lp.num_col + lp.num_row),
        SparseMatrix(lp.num_col, lp.num_row), kNotset, false, 0.0, 0, 0,
        Set{UInt64}(), -kHighsIInf, BadBasisChange[], RayRecord(), false,
        SimplexBasis(lp.num_col, lp.num_row), false,
        HVector(lp.num_row), HVector(lp.num_row), HVector(lp.num_col),
        zeros(Int, lp.num_row), zeros(50), zeros(Int, 50),
        zeros(Bool, lp.num_row), zeros(lp.num_col + lp.num_row),
        HighsRandom(1))
    # `initialiseEkk` : options internes, RNG ré-initialisé, puis un premier
    # tirage des vecteurs aléatoires (`initialiseSimplexLpRandomVectors`).
    set_simplex_options!(engine)
    initialise_simplex_lp_random_vectors!(engine)
    return engine
end

"""
`HEkk::setSimplexOptions` et `updateSimplexOptions` — recopie statique des
options dans l'espace de travail (le port ne modifie pas les options en cours
de solve).
"""
function set_simplex_options!(e::SimplexEngine)
    info = e.info
    info.simplex_strategy = kSimplexStrategyDualPlain
    info.primal_simplex_bound_perturbation_multiplier =
        e.options.primal_simplex_bound_perturbation_multiplier
    info.dual_simplex_cost_perturbation_multiplier =
        e.options.dual_simplex_cost_perturbation_multiplier
    info.dual_edge_weight_strategy = e.options.simplex_dual_edge_weight_strategy
    info.price_strategy = e.options.simplex_price_strategy
    # Les multiplicateurs de perturbation sont recopiés dans `info` comme dans
    # la source : les chemins de nettoyage croisés (dual → primal, primal →
    # dual) les mettent à zéro le temps d'un solve.
    info.factor_pivot_threshold = kDefaultPivotThreshold
    info.update_limit = e.options.simplex_update_limit
    return e
end

"""
    repaired_bounds(lower, upper, tolerance)

Lambda `infeasibleBoundOk` de `Highs::infeasibleBoundsOk` : pour des bornes
incohérentes (`lower > upper`) dont l'écart est sous la tolérance de
faisabilité primale, la borne entière (au sens `x == round(x)`) est conservée
et l'autre suit, sinon les deux sont ramenées à mi-chemin. Rend
`(ok, lower, upper)`.
"""
function repaired_bounds(lower::Float64, upper::Float64, tolerance::Float64)
    (upper - lower) > -tolerance || return false, lower, upper
    integer_lower = lower == floor(lower + 0.5)
    integer_upper = upper == floor(upper + 0.5)
    if integer_lower
        return true, lower, lower
    elseif integer_upper
        return true, upper, upper
    end
    mid = 0.5 * (lower + upper)
    return true, mid, mid
end

"""
    infeasible_bounds_ok!(e)

`Highs::infeasibleBoundsOk` (sans intégralité ni rapport) : répare en place les
bornes incohérentes sous tolérance et compte les incohérences significatives.
Rend `false` s'il en reste : `Highs::run` déclare alors `kInfeasible` sans
résoudre. Les lignes infectées sont comptées comme les colonnes (la source
ignore la valeur de retour du lambda, mais compte par effet de bord).
"""
function infeasible_bounds_ok!(e::SimplexEngine)
    lp = e.lp
    tolerance = e.options.primal_feasibility_tolerance
    num_true_infeasible_bound = 0
    for iCol ∈ 1:lp.num_col
        lower = lp.col_lower[iCol]
        upper = lp.col_upper[iCol]
        lower > upper || continue
        ok, lower, upper = repaired_bounds(lower, upper, tolerance)
        if ok
            lp.col_lower[iCol] = lower
            lp.col_upper[iCol] = upper
        else
            num_true_infeasible_bound += 1
        end
    end
    for iRow ∈ 1:lp.num_row
        lower = lp.row_lower[iRow]
        upper = lp.row_upper[iRow]
        lower > upper || continue
        ok, lower, upper = repaired_bounds(lower, upper, tolerance)
        if ok
            lp.row_lower[iRow] = lower
            lp.row_upper[iRow] = upper
        else
            num_true_infeasible_bound += 1
        end
    end
    return num_true_infeasible_bound == 0
end

"""
    initialise_for_solve!(e)

`HEkk::initialiseForSolve` (sous-ensemble M3b) : options internes, vecteurs
aléatoires (consommés comme la source), base logique et facteur si absents,
vue rowwise, coûts/bornes/valeurs, primal, dual, infaisabilités et objectifs.
Le statut passe à `kOptimal` si le point de départ est déjà primal et dual
faisable, et à `kInfeasible` (sans rien initialiser) si `Highs::run` rejette
les bornes.
"""
function initialise_for_solve!(e::SimplexEngine)
    # `HApp::solveLpSimplex` : le compteur de travail du solve repart de
    # `highs_info.simplex_iteration_count` (zéro au début de chaque `Highs_run`)
    # — c'est ce qui rend `simplex_iteration_limit` **par solve**.
    e.iteration_count0 = e.iteration_count
    # `Highs::run` : les bornes incohérentes sont réparées ou rendent le LP
    # infaisable AVANT que le simplexe ne soit touché (ni options, ni RNG).
    if !infeasible_bounds_ok!(e)
        e.bounds_infeasible = true
        e.model_status = kInfeasible
        return e
    end
    e.bounds_infeasible = false
    # `HApp::solveLpSimplex` : échelles LP avant `moveLp` (le LP peut être
    # ré-échelonné, les facteurs connus ré-appliqués, ou retirés).
    consider_scaling!(e)
    # `HEkk::setNlaPointersForLpAndScale` : les conversions NLA ne servent
    # qu'à pontifier un LP NON échelonné dont le facteur est échelonné. Ici
    # le LP est échelonné et le facteur aussi : les échelles sont déjà dans
    # le modèle (et dans `apply_lp_scale!`), les conversions doivent être
    # inactives (`scale_ == NULL`).
    e.nla.scale = e.lp.scale.has_scaling && !e.lp.is_scaled ? e.lp.scale :
                  nothing
    set_simplex_options!(e)
    # `HEkk::solve` : `initialiseControl` avant `initialiseForSolve` (le dual
    # l'appelle aussi dans `solve!`, sans effet entre-temps).
    initialise_control!(e)
    initialise_simplex_lp_random_vectors!(e)
    if !e.status.has_basis
        if e.highs_basis_valid
            # `HApp::solveLpSimplex` : base du Ekk jetée mais `basis_` valide
            # (changement de coefficient) — `HEkk::setBasis`.
            restore_basis!(e, e.highs_basis)
        else
            set_basis!(e)
        end
    end
    if !e.status.has_invert
        rank_deficiency = compute_factor!(e)
        if rank_deficiency != 0
            # Base de départ singulière : le facteur l'a complétée par des
            # logiques ; on synchronise la base et on poursuit (comme
            # `initialiseSimplexLpBasisAndFactor`).
            handle_rank_deficiency!(e)
            set_nonbasic_move!(e)
            e.status.has_basis = true
            e.status.has_invert = true
            e.status.has_fresh_invert = true
        end
    end
    initialise_partitioned_rowwise_matrix!(e)
    initialise_cost!(e, kPrimal, kSolvePhaseUnknown)
    initialise_bound!(e, kPrimal, kSolvePhaseUnknown)
    initialise_nonbasic_value_and_move!(e)
    compute_primal!(e)
    compute_dual!(e)
    compute_simplex_infeasible!(e)
    compute_dual_objective_value!(e)
    compute_primal_objective_value!(e)
    # Comme `HEkk::initialiseForSolve` : le statut est remis à `kNotset` avant
    # le test d'optimalité (sinon un `kOptimal` d'une résolution précédente
    # survit à un modèle modifié et le solve est sauté).
    e.model_status = kNotset
    if e.info.num_primal_infeasibilities == 0 &&
       e.info.num_dual_infeasibilities == 0
        e.model_status = kOptimal
    end
    empty!(e.visited_basis)
    push!(e.visited_basis, e.basis.hash)
    e.previous_iteration_cycling_detected = -kHighsIInf
    return e
end

##############################################################################
# Modifications du LP entre deux résolutions (M5)
#
# Contrat de la source : les modifications passent par le LP (`HighsLp`) puis
# `HEkk::updateStatus(LpAction)` invalide ce qui doit l'être. Un changement de
# borne ou de coût **conserve la base** (warm start), un changement de
# coefficient efface tout l'état (`HEkk::clear`, la prochaine résolution
# repart de la base logique).
##############################################################################

"""`HEkk::invalidateBasisArtifacts` — jette base, facteur et caches associés."""
function invalidate_basis_artifacts!(e::SimplexEngine)
    e.status.has_ar_matrix = false
    e.status.has_dual_steepest_edge_weights = false
    e.status.has_invert = false
    e.status.has_fresh_invert = false
    e.status.has_fresh_rebuild = false
    e.status.has_dual_objective_value = false
    e.status.has_primal_objective_value = false
    clear_ray_records!(e)
    return e
end

"""`HEkk::invalidateBasis`."""
function invalidate_basis!(e::SimplexEngine)
    e.status.has_basis = false
    invalidate_basis_artifacts!(e)
    return e
end

"""`HEkk::invalidateBasisMatrix`."""
function invalidate_basis_matrix!(e::SimplexEngine)
    invalidate_basis!(e)
    return e
end

"""
`HEkk::clear` (sans le LP, qui reste le modèle muté) : tout l'état du simplexe
est jeté, la prochaine `initialise_for_solve!` repart de la base logique.
"""
function clear_ekk!(e::SimplexEngine)
    invalidate_basis_matrix!(e)
    empty!(e.bad_basis_change)
    empty!(e.visited_basis)
    e.previous_iteration_cycling_detected = -kHighsIInf
    return e
end

"""
    update_status!(e, action)

`HEkk::updateStatus` : conséquences d'une modification du LP.

- `kLpActionScale` : la base et le NLA deviennent caducs (l'espace change) ;
- `kLpActionNewCosts`, `kLpActionNewBounds` : la base est conservée (warm
  start), mais le rebuild et les objectifs sont à refaire ;
- `kLpActionNewBasis` : la base est jetée ;
- `kLpActionNewRows` : tout l'état est jeté (changement de coefficient).
"""
function update_status!(e::SimplexEngine, action::Int)
    if action == kLpActionScale
        invalidate_basis_matrix!(e)
    elseif action == kLpActionNewCosts || action == kLpActionNewBounds
        e.status.has_fresh_rebuild = false
        e.status.has_dual_objective_value = false
        e.status.has_primal_objective_value = false
    elseif action == kLpActionNewBasis
        invalidate_basis!(e)
    elseif action == kLpActionNewRows
        clear_ekk!(e)
    else
        error("SimplexEngine : action LP $action non portée (M5)")
    end
    return e
end

# Les bornes incohérentes ne sont pas refusées ici : `assessBounds` de la
# source se contente d'un avertissement et laisse le LP les porter ;
# `infeasible_bounds_ok!` les répare ou rend le LP infaisable à la résolution.
# Seul le clamp `infinite_bound` (1e30) d'`assessBounds` n'est pas porté.

"""
    change_col_bounds!(e::SimplexEngine, iCol::Int, lower::Float64, upper::Float64)

Update lower and upper bounds of a single column variable `iCol` (1-based) in-place.
Signals the engine that bounds have changed while keeping the basis intact for warm-starting.
"""
function change_col_bounds!(e::SimplexEngine, iCol::Int, lower::Float64,
    upper::Float64)
    (1 <= iCol <= e.lp.num_col) || throw(ArgumentError("column $iCol out of bounds"))
    e.lp.col_lower[iCol] = lower
    e.lp.col_upper[iCol] = upper
    update_status!(e, kLpActionNewBounds)
    return e
end

"""
    change_cols_bounds!(e::SimplexEngine, iCols, lowers, uppers)

Update bounds for a collection of columns in-place.
"""
function change_cols_bounds!(e::SimplexEngine, iCols::AbstractVector{Int},
    lowers::AbstractVector{Float64}, uppers::AbstractVector{Float64})
    (length(iCols) == length(lowers) == length(uppers)) ||
        throw(ArgumentError("dimension mismatch in change_cols_bounds!"))
    for k ∈ eachindex(iCols)
        iCol = iCols[k]
        (1 <= iCol <= e.lp.num_col) ||
            throw(ArgumentError("column $iCol out of bounds"))
        e.lp.col_lower[iCol] = lowers[k]
        e.lp.col_upper[iCol] = uppers[k]
    end
    update_status!(e, kLpActionNewBounds)
    return e
end

"""
    change_row_bounds!(e::SimplexEngine, iRow::Int, lower::Float64, upper::Float64)

Update lower and upper bounds of a single constraint row `iRow` (1-based) in-place.
"""
function change_row_bounds!(e::SimplexEngine, iRow::Int, lower::Float64,
    upper::Float64)
    (1 <= iRow <= e.lp.num_row) || throw(ArgumentError("row $iRow out of bounds"))
    e.lp.row_lower[iRow] = lower
    e.lp.row_upper[iRow] = upper
    update_status!(e, kLpActionNewBounds)
    return e
end

"""
    change_rows_bounds!(e::SimplexEngine, iRows, lowers, uppers)

Update bounds for a collection of constraint rows in-place.
"""
function change_rows_bounds!(e::SimplexEngine, iRows::AbstractVector{Int},
    lowers::AbstractVector{Float64}, uppers::AbstractVector{Float64})
    (length(iRows) == length(lowers) == length(uppers)) ||
        throw(ArgumentError("dimension mismatch in change_rows_bounds!"))
    for k ∈ eachindex(iRows)
        iRow = iRows[k]
        (1 <= iRow <= e.lp.num_row) ||
            throw(ArgumentError("row $iRow out of bounds"))
        e.lp.row_lower[iRow] = lowers[k]
        e.lp.row_upper[iRow] = uppers[k]
    end
    update_status!(e, kLpActionNewBounds)
    return e
end

"""
    change_cols_cost!(e::SimplexEngine, iCols, costs)

Update objective cost coefficients for a collection of columns in-place.
"""
function change_cols_cost!(e::SimplexEngine, iCols::AbstractVector{Int},
    costs::AbstractVector{Float64})
    length(iCols) == length(costs) ||
        throw(ArgumentError("dimension mismatch in change_cols_cost!"))
    for k ∈ eachindex(iCols)
        iCol = iCols[k]
        (1 <= iCol <= e.lp.num_col) ||
            throw(ArgumentError("column $iCol out of bounds"))
        e.lp.col_cost[iCol] = costs[k]
    end
    update_status!(e, kLpActionNewCosts)
    return e
end

"""`Highs_changeObjectiveSense`."""
function change_objective_sense!(e::SimplexEngine, sense::ObjSense)
    e.lp.sense == sense && return e
    e.lp.sense = sense
    update_status!(e, kLpActionNewCosts)
    return e
end

"""
    change_coeff!(e, iRow, iCol, value)

`Highs_changeCoeff` : un coefficient de la matrice (CSC). Un coefficient
inférieur ou égal à `small_matrix_value` est traité comme nul : il supprime
l'entrée existante, et une entrée absente n'est pas créée. Un vrai changement
invalide tout l'état (`HEkk::updateStatus(kNewRows)` → `clear`).
"""
function change_coeff!(e::SimplexEngine, iRow::Int, iCol::Int, value::Float64)
    lp = e.lp
    (1 <= iRow <= lp.num_row && 1 <= iCol <= lp.num_col) ||
        throw(ArgumentError("coefficient ($iRow, $iCol) hors du LP"))
    m = lp.a_matrix
    is_colwise(m) || error("change_coeff! exige la vue colwise")
    zero_new_value = abs(value) <= e.options.small_matrix_value
    change_el = 0
    for iEl ∈ m.start[iCol]:(m.start[iCol + 1] - 1)
        if m.index[iEl] == iRow
            change_el = iEl
            break
        end
    end
    if change_el == 0
        # Pas de non nul existant : un petit coefficient est ignoré.
        if !zero_new_value
            insert!(m.index, m.start[iCol + 1], iRow)
            insert!(m.value, m.start[iCol + 1], value)
            for i ∈ (iCol + 1):(lp.num_col + 1)
                m.start[i] += 1
            end
        end
    elseif zero_new_value
        # Le coefficient annule un non nul existant : le retirer.
        deleteat!(m.index, change_el)
        deleteat!(m.value, change_el)
        for i ∈ (iCol + 1):(lp.num_col + 1)
            m.start[i] -= 1
        end
    else
        m.index[change_el] = iRow
        m.value[change_el] = value
    end
    # La source invalide l'état même si le coefficient n'a pas bougé.
    update_status!(e, kLpActionNewRows)
    return e
end

"""
`HEkk::initialiseSimplexLpRandomVectors` : permutations aléatoires des indices
et `numTotRandomValue`. L'état du RNG est consommé dans l'ordre de la source
(colonnes, puis toutes les variables, puis les réels) : c'est lui qui fixe
l'ordre de balayage de CHUZR.
"""
function initialise_simplex_lp_random_vectors!(e::SimplexEngine)
    num_col = e.lp.num_col
    num_tot = num_col + e.lp.num_row
    num_tot == 0 && return e
    if num_col > 0
        num_col_permutation = e.info.numColPermutation
        resize!(num_col_permutation, num_col)
        for i ∈ 1:num_col
            num_col_permutation[i] = i
        end
        shuffle!(e.random, num_col_permutation)
    end
    num_tot_permutation = e.info.numTotPermutation
    resize!(num_tot_permutation, num_tot)
    for i ∈ 1:num_tot
        num_tot_permutation[i] = i
    end
    shuffle!(e.random, num_tot_permutation)
    num_tot_random_value = e.info.numTotRandomValue
    resize!(num_tot_random_value, num_tot)
    for i ∈ 1:num_tot
        num_tot_random_value[i] = fraction(e.random)
    end
    return e
end

"""
Mouvement non basique déduit des seules bornes — corps commun de
`HEkk::setBasis` et `HEkk::setNonbasicMove`. La branche `have_solution` de
`setNonbasicMove` est constante (`false`) dans la source : elle n'est pas
portée. Toute combinaison de bornes est couverte, `kIllegalMoveValue` ne peut
pas être rendu.
"""
function nonbasic_move_from_bounds(lower::Float64, upper::Float64)
    if lower == upper
        return kNonbasicMoveZe                      # fixe
    elseif lower > -kHighsInf
        if upper < kHighsInf
            # Boxée : borne la plus proche de zéro (le C++ compare |lower|<|upper|).
            return abs(lower) < abs(upper) ? kNonbasicMoveUp : kNonbasicMoveDn
        end
        return kNonbasicMoveUp                      # minorée
    elseif upper < kHighsInf
        return kNonbasicMoveDn                      # majorée
    end
    return kNonbasicMoveZe                          # libre
end

"""
    set_basis!(e)

Base logique (`HEkk::setBasis`) : colonnes non basiques, logiques basiques
(`basicIndex[iRow] = num_col + iRow`). Le `hash` de la source n'est pas porté.
"""
function set_basis!(e::SimplexEngine)
    lp, basis = e.lp, e.basis
    setup!(basis, lp.num_col, lp.num_row)
    for iCol ∈ 1:lp.num_col
        basis.nonbasicFlag[iCol] = kNonbasicFlagTrue
        basis.nonbasicMove[iCol] = nonbasic_move_from_bounds(lp.col_lower[iCol],
            lp.col_upper[iCol])
    end
    for iRow ∈ 1:lp.num_row
        iVar = lp.num_col + iRow
        basis.nonbasicFlag[iVar] = kNonbasicFlagFalse
        basis.basicIndex[iRow] = iVar
        basis.hash = sparse_combine(basis.hash, iVar - 1)
    end
    e.info.num_basic_logicals = lp.num_row
    e.status.has_basis = true
    return e
end

"""
    handle_rank_deficiency!(e)

`HEkk::handleRankDeficiency` : le facteur a déjà complété une base singulière
avec des variables logiques (`buildHandleRankDeficiency`/`buildMarkSingC`) ;
on synchronise les drapeaux, on rend le changement tabou (raison `kSingular`)
et on invalide la vue rowwise.
"""
function handle_rank_deficiency!(e::SimplexEngine)
    factor = e.nla.factor
    for k ∈ 1:factor.rank_deficiency
        row_in = factor.row_with_no_pivot[k]
        variable_in = e.lp.num_col + row_in
        variable_out = factor.var_with_no_pivot[k]
        e.basis.nonbasicFlag[variable_in] = kNonbasicFlagFalse
        variable_out > 0 &&
            (e.basis.nonbasicFlag[variable_out] = kNonbasicFlagTrue)
        add_bad_basis_change!(e, row_in, variable_in, variable_out,
            kBadBasisChangeSingular, true)
    end
    e.status.has_ar_matrix = false
    e.status.has_dual_steepest_edge_weights = false
    return e
end

"""`HEkk::logicalBasis` — toutes les variables de base sont des logiques."""
function logical_basis(e::SimplexEngine)
    return all(iRow -> e.basis.basicIndex[iRow] > e.lp.num_col,
        1:e.lp.num_row)
end

"""
`HEkk::setBasis(const HighsBasis&)` : installe `from` dans la base partagée
(recopie en place, `basicIndex` reste le tableau du facteur et du NLA).
"""
function restore_basis!(e::SimplexEngine, from::SimplexBasis)
    basis = e.basis
    length(basis.basicIndex) == length(from.basicIndex) ||
        throw(ArgumentError("base de tailles différentes"))
    copyto!(basis.basicIndex, from.basicIndex)
    copyto!(basis.nonbasicFlag, from.nonbasicFlag)
    copyto!(basis.nonbasicMove, from.nonbasicMove)
    basis.hash = from.hash
    e.status.has_basis = true
    return e
end

"""
`HApp::solveLpSimplex` (fin de résolution) : `basis_ = getHighsBasis(ekk)` —
la base rendue devient la base mémorisée du niveau Highs, utilisée au prochain
solve si celle du Ekk a été jetée.
"""
function store_solution_basis!(e::SimplexEngine)
    if e.status.has_basis
        copy_basis!(e.highs_basis, e.basis)
        e.highs_basis_valid = true
    end
    return e
end

"""`HEkk::clearRayRecords` — oublie le rayon primal (`clear` de `HighsRayRecord`)."""
function clear_ray_records!(e::SimplexEngine)
    clear!(e.primal_ray_record)
    return e
end

"""Hachage d'une base quelconque : combinaison des variables basiques."""
function basis_hash(basis::SimplexBasis)
    hash = UInt64(0)
    for iVar ∈ eachindex(basis.nonbasicFlag)
        if basis.nonbasicFlag[iVar] == kNonbasicFlagFalse
            hash = sparse_combine(hash, iVar - 1)
        end
    end
    return hash
end

"""
    set_nonbasic_move!(e)

`HEkk::setNonbasicMove` : recalcule `nonbasicMove` de toutes les variables à
partir des bornes du **LP** (pas celles de l'espace de travail), sans toucher
aux valeurs. Une variable basique reçoit `kNonbasicMoveZe`.
"""
function set_nonbasic_move!(e::SimplexEngine)
    lp, basis = e.lp, e.basis
    num_col = lp.num_col
    @inbounds for iVar ∈ 1:(num_col + lp.num_row)
        if basis.nonbasicFlag[iVar] == kNonbasicFlagFalse
            basis.nonbasicMove[iVar] = kNonbasicMoveZe
            continue
        end
        if iVar <= num_col
            lower, upper = lp.col_lower[iVar], lp.col_upper[iVar]
        else
            iRow = iVar - num_col
            lower, upper = -lp.row_upper[iRow], -lp.row_lower[iRow]
        end
        basis.nonbasicMove[iVar] = nonbasic_move_from_bounds(lower, upper)
    end
    return e
end

"""`HEkk::initialiseLpColBound` — bornes colonnes de l'espace de travail."""
function initialise_lp_col_bound!(e::SimplexEngine)
    lp, info = e.lp, e.info
    @inbounds for iCol ∈ 1:lp.num_col
        info.workLower[iCol] = lp.col_lower[iCol]
        info.workUpper[iCol] = lp.col_upper[iCol]
        info.workRange[iCol] = info.workUpper[iCol] - info.workLower[iCol]
        info.workLowerShift[iCol] = 0.0
        info.workUpperShift[iCol] = 0.0
    end
    return e
end

"""`HEkk::initialiseLpRowBound` — bornes logiques `[-row_upper, -row_lower]`."""
function initialise_lp_row_bound!(e::SimplexEngine)
    lp, info = e.lp, e.info
    num_col = lp.num_col
    @inbounds for iRow ∈ 1:lp.num_row
        iVar = num_col + iRow
        info.workLower[iVar] = -lp.row_upper[iRow]
        info.workUpper[iVar] = -lp.row_lower[iRow]
        info.workRange[iVar] = info.workUpper[iVar] - info.workLower[iVar]
        info.workLowerShift[iVar] = 0.0
        info.workUpperShift[iVar] = 0.0
    end
    return e
end

"""
    initialise_bound!(e, algorithm, solve_phase; perturb = false)

`HEkk::initialiseBound` : bornes de l'espace de travail depuis le LP, puis,
pour le simplexe dual hors phase 2, bornes spéciales
`[-1000, 1000]`/`[-1, 0]`/`[0, 1]`/`[0, 0]` qui font de l'objectif dual
l'opposé de la somme des infaisabilités. Pour le simplexe primal, la
perturbation aléatoire des bornes (base `multiplicateur · 5e-7`, relative à la
borne, absolue si `|borne| < 1`) écarte les bornes finies ; une variable fixe
non basique est laissée intacte.
"""
function initialise_bound!(e::SimplexEngine, algorithm::SimplexAlgorithm,
    solve_phase::Int; perturb::Bool=false)
    initialise_lp_col_bound!(e)
    initialise_lp_row_bound!(e)
    info = e.info
    info.bounds_shifted = false
    info.bounds_perturbed = false
    if algorithm == kPrimal
        (!perturb || info.primal_simplex_bound_perturbation_multiplier == 0) &&
            return e
        base = info.primal_simplex_bound_perturbation_multiplier * 5e-7
        for iVar ∈ 1:(e.lp.num_col + e.lp.num_row)
            lower = info.workLower[iVar]
            upper = info.workUpper[iVar]
            # Une variable fixe non basique reste à sa borne : sa borne n'est
            # pas perturbée (elle ne peut pas bouger).
            if e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue && lower == upper
                continue
            end
            random_value = info.numTotRandomValue[iVar]
            if lower > -kHighsInf
                if lower < -1
                    lower -= random_value * base * (-lower)
                elseif lower < 1
                    lower -= random_value * base
                else
                    lower -= random_value * base * lower
                end
                info.workLower[iVar] = lower
            end
            if upper < kHighsInf
                if upper < -1
                    upper += random_value * base * (-upper)
                elseif upper < 1
                    upper += random_value * base
                else
                    upper += random_value * base * upper
                end
                info.workUpper[iVar] = upper
            end
            info.workRange[iVar] = info.workUpper[iVar] - info.workLower[iVar]
            if e.basis.nonbasicFlag[iVar] == kNonbasicFlagFalse
                continue
            end
            if e.basis.nonbasicMove[iVar] > 0
                info.workValue[iVar] = lower
            elseif e.basis.nonbasicMove[iVar] < 0
                info.workValue[iVar] = upper
            end
        end
        for iRow ∈ 1:e.lp.num_row
            iVar = e.basis.basicIndex[iRow]
            info.baseLower[iRow] = info.workLower[iVar]
            info.baseUpper[iRow] = info.workUpper[iVar]
        end
        info.bounds_perturbed = true
        return e
    end
    solve_phase == kSolvePhase2 && return e
    for iVar ∈ 1:(e.lp.num_col + e.lp.num_row)
        if info.workLower[iVar] == -kHighsInf && info.workUpper[iVar] == kHighsInf
            info.workLower[iVar] = -1000.0          # libre
            info.workUpper[iVar] = 1000.0
        elseif info.workLower[iVar] == -kHighsInf
            info.workLower[iVar] = -1.0             # majorée
            info.workUpper[iVar] = 0.0
        elseif info.workUpper[iVar] == kHighsInf
            info.workLower[iVar] = 0.0              # minorée
            info.workUpper[iVar] = 1.0
        else
            info.workLower[iVar] = 0.0              # boxée ou fixe
            info.workUpper[iVar] = 0.0
        end
        info.workRange[iVar] = info.workUpper[iVar] - info.workLower[iVar]
    end
    return e
end

"""
`HEkk::initialiseLpColCost` — coûts signés (`sense`) et mis à l'échelle par
`2^cost_scale_factor` ; les shifts sont remis à zéro.
"""
function initialise_lp_col_cost!(e::SimplexEngine)
    cost_scale_factor = 2.0^e.options.cost_scale_factor
    sense_scaled = Int(e.lp.sense) * cost_scale_factor
    col_cost = e.lp.col_cost
    workCost = e.info.workCost
    workShift = e.info.workShift
    @inbounds for iCol ∈ 1:e.lp.num_col
        workCost[iCol] = sense_scaled * col_cost[iCol]
        workShift[iCol] = 0.0
    end
    return e
end

"""`HEkk::initialiseLpRowCost` — coût nul pour les variables logiques."""
function initialise_lp_row_cost!(e::SimplexEngine)
    workCost = e.info.workCost
    workShift = e.info.workShift
    @inbounds for iVar ∈ (e.lp.num_col + 1):(e.lp.num_col + e.lp.num_row)
        workCost[iVar] = 0.0
        workShift[iVar] = 0.0
    end
    return e
end

"""
    initialise_cost!(e, algorithm, solve_phase; perturb = false)

`HEkk::initialiseCost` : copie les coûts du LP (le `solve_phase` n'entre pas
dans le calcul, comme dans la source), puis, si `perturb` est demandé et que le
multiplicateur est non nul, applique la perturbation aléatoire duale :
`xpert = (1 + r_i) (|c_i| + 1) · base` où
`base = multiplicateur · 5e-7 · max|c|` (réduit par `sqrt(sqrt(·))` au-delà de
100, et plafonné à 1 si moins de 1 % des variables sont boxées) ; les coûts
logiques reçoivent `(0.5 - r_i) · multiplicateur · 1e-12`.
Le bloc de statistiques couplé à `output_flag` n'est pas porté : il n'est pas
atteignable depuis le dual (coûts tous nuls ⇒ `force_phase2` ⇒ pas de
perturbation).
"""
function initialise_cost!(e::SimplexEngine, algorithm::SimplexAlgorithm,
    solve_phase::Int; perturb::Bool=false)
    initialise_lp_col_cost!(e)
    initialise_lp_row_cost!(e)
    info = e.info
    info.costs_shifted = false
    info.costs_perturbed = false
    algorithm == kPrimal && return e
    multiplier = info.dual_simplex_cost_perturbation_multiplier
    (!perturb || multiplier == 0) && return e
    max_abs_cost = 0.0
    for i ∈ 1:e.lp.num_col
        max_abs_cost = max(max_abs_cost, abs(info.workCost[i]))
    end
    if max_abs_cost > 100.0
        max_abs_cost = sqrt(sqrt(max_abs_cost))
    end
    num_tot = e.lp.num_col + e.lp.num_row
    boxed_rate = 0.0
    for i ∈ 1:num_tot
        boxed_rate += info.workRange[i] < 1e30
    end
    boxed_rate /= num_tot
    if boxed_rate < 0.01
        max_abs_cost = min(max_abs_cost, 1.0)
    end
    cost_perturbation_base = multiplier * 5e-7 * max_abs_cost
    for i ∈ 1:e.lp.num_col
        lower = e.lp.col_lower[i]
        upper = e.lp.col_upper[i]
        xpert = (1 + info.numTotRandomValue[i]) *
                (abs(info.workCost[i]) + 1) * cost_perturbation_base
        if lower == -kHighsInf && upper == kHighsInf
            # libre : pas de perturbation
        elseif upper == kHighsInf
            info.workCost[i] += xpert                  # minorée
        elseif lower == -kHighsInf
            info.workCost[i] -= xpert                  # majorée
        elseif lower != upper
            info.workCost[i] += info.workCost[i] >= 0 ? xpert : -xpert  # boxée
        end
    end
    row_cost_perturbation_base = multiplier * 1e-12
    for i ∈ (e.lp.num_col + 1):num_tot
        info.workCost[i] +=
            (0.5 - info.numTotRandomValue[i]) * row_cost_perturbation_base
    end
    info.costs_perturbed = true
    return e
end

"""
    initialise_nonbasic_value_and_move!(e)

`HEkk::initialiseNonbasicValueAndMove` : valeurs et mouvements non basiques
depuis `nonbasicFlag` et les bornes de l'espace de travail. Une boxée garde le
côté désigné par son `nonbasicMove` d'origine ; un mouvement invalide est
corrigé en `kNonbasicMoveUp` (comme la source). Les basiques restent à zéro :
leurs valeurs viennent de `compute_primal!`.
"""
function initialise_nonbasic_value_and_move!(e::SimplexEngine)
    basis, info = e.basis, e.info
    for iVar ∈ 1:(e.lp.num_col + e.lp.num_row)
        if basis.nonbasicFlag[iVar] == kNonbasicFlagFalse
            basis.nonbasicMove[iVar] = kNonbasicMoveZe
            continue
        end
        lower, upper = info.workLower[iVar], info.workUpper[iVar]
        original_move = basis.nonbasicMove[iVar]
        if lower == upper
            value, move = lower, kNonbasicMoveZe
        elseif lower > -kHighsInf
            if upper < kHighsInf
                if original_move == kNonbasicMoveUp
                    value, move = lower, kNonbasicMoveUp
                elseif original_move == kNonbasicMoveDn
                    value, move = upper, kNonbasicMoveDn
                else
                    value, move = lower, kNonbasicMoveUp
                end
            else
                value, move = lower, kNonbasicMoveUp
            end
        elseif upper < kHighsInf
            value, move = upper, kNonbasicMoveDn
        else
            value, move = 0.0, kNonbasicMoveZe
        end
        basis.nonbasicMove[iVar] = move
        info.workValue[iVar] = value
    end
    return e
end

"""`HEkk::updateOperationResultDensity` — moyenne glissante des densités."""
update_operation_result_density(density::Float64, local_density::Float64) =
    (1 - kRunningAverageMultiplier) * density +
    kRunningAverageMultiplier * local_density

"""`HEkk::fullBtran` — BTRAN complet, puis moyenne glissante `dual_col_density`."""
function full_btran!(e::SimplexEngine, buffer::HVector)
    btran!(e.nla, buffer, e.info.dual_col_density)
    e.info.dual_col_density = update_operation_result_density(
        e.info.dual_col_density, buffer.count / e.lp.num_row)
    return buffer
end

"""`HEkk::fullPrice` — `full_row = A^T full_col`, prix colonne par colonne."""
function full_price!(e::SimplexEngine, full_col::HVector, full_row::HVector)
    clear!(full_row)
    price_by_column!(e.lp.a_matrix, full_row, full_col)
    return full_row
end

"""
    compute_primal!(e)

`HEkk::computePrimal` : `x_B = -B^{-1} N x_N` par `collectAj` puis FTRAN, dans
le tampon local de la source (routine d'initialisation, hors du pivot courant).
`baseValue` suit l'ordre des colonnes de base, `baseLower`/`baseUpper` sont
recopiés, et les compteurs d'infaisabilité primal sont invalidés.
"""
function compute_primal!(e::SimplexEngine)
    num_row, num_col = e.lp.num_row, e.lp.num_col
    primal_col = e.primal_col
    clear!(primal_col)
    @inbounds for iVar ∈ 1:(num_col + num_row)
        if e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue &&
           e.info.workValue[iVar] != 0.0
            collect_aj!(e.lp.a_matrix, primal_col, iVar, e.info.workValue[iVar])
        end
    end
    if primal_col.count != 0
        ftran!(e.nla, primal_col, e.info.primal_col_density)
        e.info.primal_col_density = update_operation_result_density(
            e.info.primal_col_density, primal_col.count / num_row)
    end
    @inbounds @simd for iRow ∈ 1:num_row
        e.info.baseValue[iRow] = -primal_col.array[iRow]
    end
    @inbounds for iRow ∈ 1:num_row
        iVar = e.basis.basicIndex[iRow]
        e.info.baseLower[iRow] = e.info.workLower[iVar]
        e.info.baseUpper[iRow] = e.info.workUpper[iVar]
    end
    e.info.num_primal_infeasibilities = kHighsIllegalInfeasibilityCount
    e.info.max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure
    e.info.sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure
    return e
end

"""
    compute_dual!(e)

`HEkk::computeDual` : `workDual = workCost + workShift - [A I]^T pi` avec
`pi = B^{-T} c_B` (BTRAN complet sur les coûts basiques), `A^T pi` par
`priceByColumn`. Les valeurs duales des basiques sont nulles à l'arrondi. Les
compteurs d'infaisabilité duale sont invalidés.
"""
function compute_dual!(e::SimplexEngine)
    num_col, num_row = e.lp.num_col, e.lp.num_row
    num_tot = num_col + num_row
    dual_col = e.dual_col
    clear!(dual_col)
    @inbounds for iRow ∈ 1:num_row
        iVar = e.basis.basicIndex[iRow]
        value = e.info.workCost[iVar] + e.info.workShift[iVar]
        if value != 0.0
            dual_col.count += 1
            dual_col.index[dual_col.count] = iRow
            dual_col.array[iRow] = value
        end
    end
    @inbounds @simd for iVar ∈ 1:num_tot
        e.info.workDual[iVar] = e.info.workCost[iVar] + e.info.workShift[iVar]
    end
    if dual_col.count != 0
        full_btran!(e, dual_col)
        dual_row = e.dual_row
        full_price!(e, dual_col, dual_row)
        @inbounds @simd for iCol ∈ 1:num_col
            e.info.workDual[iCol] -= dual_row.array[iCol]
        end
        @inbounds @simd for iVar ∈ (num_col + 1):num_tot
            e.info.workDual[iVar] -= dual_col.array[iVar - num_col]
        end
    end
    e.info.num_dual_infeasibilities = kHighsIllegalInfeasibilityCount
    e.info.max_dual_infeasibility = kHighsIllegalInfeasibilityMeasure
    e.info.sum_dual_infeasibilities = kHighsIllegalInfeasibilityMeasure
    return e
end

"""
    compute_primal_objective_value!(e)

`HEkk::computePrimalObjectiveValue` :
`cost_scale * (c_B^T x_B + c_N^T x_N) + offset`. Les coûts d'origine servent
au calcul (pas `workCost`) et seules les variables structurelles contribuent.
Rend la valeur, également stockée dans `info.primal_objective_value`.
"""
function compute_primal_objective_value!(e::SimplexEngine)
    value = 0.0
    @inbounds for iRow ∈ 1:e.lp.num_row
        iVar = e.basis.basicIndex[iRow]
        if iVar <= e.lp.num_col
            value += e.info.baseValue[iRow] * e.lp.col_cost[iVar]
        end
    end
    @inbounds for iCol ∈ 1:e.lp.num_col
        if e.basis.nonbasicFlag[iCol] == kNonbasicFlagTrue
            value += e.info.workValue[iCol] * e.lp.col_cost[iCol]
        end
    end
    value *= e.cost_scale
    value += e.lp.offset
    e.info.primal_objective_value = value
    e.status.has_primal_objective_value = true
    return value
end

"""
    compute_dual_objective_value!(e, phase = kSolvePhase2)

`HEkk::computeDualObjectiveValue` : `cost_scale * Σ x_i workDual_i` sur les
non basiques, plus `sense * offset` sauf en phase 1 (« l'objectif dual n'a pas
de décalage »). Rend la valeur, également stockée dans
`info.dual_objective_value`.
"""
function compute_dual_objective_value!(e::SimplexEngine,
    phase::Int=kSolvePhase2)
    value = 0.0
    @inbounds for iVar ∈ 1:(e.lp.num_col + e.lp.num_row)
        if e.basis.nonbasicFlag[iVar] == kNonbasicFlagTrue
            value += e.info.workValue[iVar] * e.info.workDual[iVar]
        end
    end
    value *= e.cost_scale
    if phase != kSolvePhase1
        value += Int(e.lp.sense) * e.lp.offset
    end
    e.info.dual_objective_value = value
    e.status.has_dual_objective_value = true
    return value
end

# --- Tranche M3b : primitives de `HEkk` pour le simplexe dual ----------------

"""
    compute_factor!(e)

`HEkk::computeFactor` : INVERT du facteur sur la base courante. Rend la carence
de rang (`0` si B^{-1} est frais). `update_count` repart à zéro et
`build_synthetic_tick` est relevé — il sert au déclenchement des
ré-inversions.
"""
function compute_factor!(e::SimplexEngine)
    rank_deficiency = invert!(e.nla)
    e.build_synthetic_tick = e.nla.build_synthetic_tick
    if rank_deficiency != 0
        e.status.has_invert = false
        e.status.has_fresh_invert = false
    else
        e.status.has_invert = true
        e.status.has_fresh_invert = true
    end
    e.info.update_count = 0
    return rank_deficiency
end

"""
    get_nonsingular_inverse!(e, solve_phase)

`HEkk::getNonsingularInverse` sans backtracking : rend `false` si la base est
singulière (la source tente alors une base de backtracking, M3c).
"""
function get_nonsingular_inverse!(e::SimplexEngine, solve_phase::Int)
    # Les poids DSE sont identifiés aux rangées : on les éparpille selon
    # `basic_index` avant INVERT, puis on les rassemble selon la permutation
    # produite à l'issue (base restaurée par backtracking le cas échéant).
    basic_index = e.basis.basicIndex
    basic_index_before = e.basic_index_before
    copyto!(basic_index_before, basic_index)
    simplex_update_count = e.info.update_count
    scattered = e.scattered_dual_edge_weight
    for i ∈ 1:e.lp.num_row
        scattered[basic_index[i]] = e.dual_edge_weight[i]
    end
    rank_deficiency = compute_factor!(e)
    if rank_deficiency != 0
        # Base singulière : retour à la dernière base non singulière.
        deficient_hash = e.basis.hash
        get_backtracking_basis!(e) || return false
        e.info.backtracking = true
        empty!(e.visited_basis)
        push!(e.visited_basis, e.basis.hash, deficient_hash)
        e.status.has_ar_matrix = false
        e.status.has_fresh_rebuild = false
        e.status.has_dual_objective_value = false
        e.status.has_primal_objective_value = false
        compute_factor!(e) == 0 || return false
        simplex_update_count <= 1 && return false
        e.info.update_limit = simplex_update_count ÷ 2
    else
        put_backtracking_basis!(e, basic_index_before)
        e.info.backtracking = false
        e.info.update_limit = e.options.simplex_update_limit
    end
    for i ∈ 1:e.lp.num_row
        e.dual_edge_weight[i] = scattered[basic_index[i]]
    end
    return true
end

"""
    put_backtracking_basis!(e)
    put_backtracking_basis!(e, basic_index_before_compute_factor)

`HEkk::putBacktrackingBasis` : sauvegarde la base courante (ou l'ordre des
colonnes de base d'avant INVERT) et l'état associé.
"""
function put_backtracking_basis!(e::SimplexEngine)
    info = e.info
    info.valid_backtracking_basis = true
    copy_basis!(info.backtracking_basis, e.basis)
    info.backtracking_basis_costs_shifted = info.costs_shifted
    info.backtracking_basis_costs_perturbed = info.costs_perturbed
    info.backtracking_basis_bounds_shifted = info.bounds_shifted
    info.backtracking_basis_bounds_perturbed = info.bounds_perturbed
    resize!(info.backtracking_basis_workShift, length(info.workShift))
    copyto!(info.backtracking_basis_workShift, info.workShift)
    resize!(info.backtracking_basis_workLowerShift, length(info.workLowerShift))
    copyto!(info.backtracking_basis_workLowerShift, info.workLowerShift)
    resize!(info.backtracking_basis_workUpperShift, length(info.workUpperShift))
    copyto!(info.backtracking_basis_workUpperShift, info.workUpperShift)
    resize!(info.backtracking_basis_edge_weight, length(e.scattered_dual_edge_weight))
    copyto!(info.backtracking_basis_edge_weight, e.scattered_dual_edge_weight)
    return e
end

function put_backtracking_basis!(e::SimplexEngine,
    basic_index_before_compute_factor::Vector{Int})
    put_backtracking_basis!(e)
    bb = e.info.backtracking_basis
    resize!(bb.basicIndex, length(basic_index_before_compute_factor))
    copyto!(bb.basicIndex, basic_index_before_compute_factor)
    return e
end

"""`HEkk::getBacktrackingBasis` — restaure la dernière base non singulière."""
function get_backtracking_basis!(e::SimplexEngine)
    info = e.info
    info.valid_backtracking_basis || return false
    copy_basis!(e.basis, info.backtracking_basis)
    info.costs_shifted = info.backtracking_basis_costs_shifted
    info.costs_perturbed = info.backtracking_basis_costs_perturbed
    info.bounds_shifted = info.backtracking_basis_bounds_shifted
    info.bounds_perturbed = info.backtracking_basis_bounds_perturbed
    resize!(info.workShift, length(info.backtracking_basis_workShift))
    copyto!(info.workShift, info.backtracking_basis_workShift)
    copyto!(e.scattered_dual_edge_weight, info.backtracking_basis_edge_weight)
    return true
end

"""
    is_bad_basis_change!(e, algorithm, variable_in, row_out, rebuild_reason)

`HEkk::isBadBasisChange` : détecte un cyclage (hachage de base déjà visité sur
itérations successives) ou un changement déjà listé comme mauvais, et le rend
tabou.
"""
function is_bad_basis_change!(e::SimplexEngine, algorithm::SimplexAlgorithm,
    variable_in::Int, row_out::Int, rebuild_reason::Int)
    rebuild_reason != kRebuildReasonNo && return false
    (variable_in == -1 || row_out == -1) && return false
    currhash = e.basis.hash
    variable_out = e.basis.basicIndex[row_out]
    currhash = sparse_inverse_combine(currhash, variable_out - 1)
    currhash = sparse_combine(currhash, variable_in - 1)
    cycling_detected = false
    possible_cycling = currhash ∈ e.visited_basis
    if possible_cycling
        if e.iteration_count == e.previous_iteration_cycling_detected + 1
            cycling_detected = true
        else
            e.previous_iteration_cycling_detected = e.iteration_count
        end
    end
    if cycling_detected
        add_bad_basis_change!(e, row_out, variable_out, variable_in,
            kBadBasisChangeCycling, true)
        return true
    end
    for change ∈ e.bad_basis_change
        if change.variable_out == variable_out &&
           change.variable_in == variable_in && change.row_out == row_out
            change.taboo = true
            return true
        end
    end
    return false
end

"""
    add_bad_basis_change!(e, row_out, variable_out, variable_in, reason, taboo)

`HEkk::addBadBasisChange` : ajoute (ou met à jour le drapeau tabou d') un
changement de base. Rend son index.
"""
function add_bad_basis_change!(e::SimplexEngine, row_out::Int,
    variable_out::Int, variable_in::Int, reason::Int, taboo::Bool)
    num_bad = length(e.bad_basis_change)
    index = -1
    for i ∈ 1:num_bad
        record = e.bad_basis_change[i]
        if record.row_out == row_out && record.variable_out == variable_out &&
           record.variable_in == variable_in && record.reason == reason
            index = i
            break
        end
    end
    if index < 0
        push!(e.bad_basis_change,
            BadBasisChange(row_out, variable_out, variable_in, reason, taboo))
        index = length(e.bad_basis_change)
    else
        e.bad_basis_change[index].taboo = taboo
    end
    return index
end

"""`HEkk::clearBadBasisChange` — vide la liste, ou seulement une raison."""
function clear_bad_basis_change!(e::SimplexEngine, reason::Int)
    if reason == kBadBasisChangeAll
        empty!(e.bad_basis_change)
    else
        filter!(change -> change.reason != reason, e.bad_basis_change)
    end
    return e
end

"""`HEkk::clearBadBasisChangeTabooFlag`."""
function clear_bad_basis_change_taboo_flag!(e::SimplexEngine)
    for change ∈ e.bad_basis_change
        change.taboo = false
    end
    return e
end

"""`HEkk::tabooBadBasisChange` — au moins un changement tabou est listé."""
taboo_bad_basis_change(e::SimplexEngine) =
    any(change -> change.taboo, e.bad_basis_change)

"""`HEkk::applyTabooRowOut` — annule l'infaisabilité des rangées taboues."""
function apply_taboo_row_out!(e::SimplexEngine, values::Vector{Float64},
    overwrite_with::Float64)
    for change ∈ e.bad_basis_change
        if change.taboo
            iRow = change.row_out
            change.save_value = values[iRow]
            values[iRow] = overwrite_with
        end
    end
    return e
end

"""`HEkk::unapplyTabooRowOut` — restaure les valeurs dans l'ordre inverse."""
function unapply_taboo_row_out!(e::SimplexEngine, values::Vector{Float64})
    for iX ∈ length(e.bad_basis_change):-1:1
        change = e.bad_basis_change[iX]
        if change.taboo
            values[change.row_out] = change.save_value
        end
    end
    return e
end

"""
`HEkk::updateBadBasisChange` : oublie les changements mauvais dont l'entrée
pivot a un effet primal **au moins** égal à la tolérance (le prédicat de la
source est `>= tolérance`, appliqué par `remove_if`).
"""
function update_bad_basis_change!(e::SimplexEngine, col_aq::HVector,
    theta_primal::Float64)
    isempty(e.bad_basis_change) && return e
    tolerance = e.options.primal_feasibility_tolerance
    filter!(change ->
        abs(col_aq.array[change.row_out] * theta_primal) < tolerance,
        e.bad_basis_change)
    return e
end

"""
    initialise_control!(e)

`HEkk::initialiseControl` : seuil de bascule DSE/Devex, compteur de contrôle,
densités remises à leurs valeurs initiales (`dual_col_density = 1`).
"""
function initialise_control!(e::SimplexEngine)
    info = e.info
    info.allow_dual_steepest_edge_to_devex_switch =
        e.options.simplex_dual_edge_weight_strategy ==
        kSimplexEdgeWeightStrategyChoose
    info.control_iteration_count0 = e.iteration_count
    info.col_aq_density = 0.0
    info.row_ep_density = 0.0
    info.row_ap_density = 0.0
    info.row_DSE_density = 0.0
    info.col_steepest_edge_density = 0.0
    info.col_BFRT_density = 0.0
    info.primal_col_density = 0.0
    info.dual_col_density = 1.0
    info.col_basic_feasibility_change_density = 0.0
    info.row_basic_feasibility_change_density = 0.0
    info.costly_DSE_frequency = 0.0
    info.num_costly_DSE_iteration = 0
    info.costly_DSE_measure = 0.0
    info.average_log_low_DSE_weight_error = 0.0
    info.average_log_high_DSE_weight_error = 0.0
    return e
end

"""`HEkk::computeDualSteepestEdgeWeights` — poids DSE de toutes les rangées."""
function compute_dual_steepest_edge_weights!(e::SimplexEngine,
    initial::Bool=false)
    row_ep = e.dual_col
    for iRow ∈ 1:e.lp.num_row
        e.dual_edge_weight[iRow] =
            compute_dual_steepest_edge_weight(e, iRow, row_ep)
    end
    return e
end

"""
`HEkk::computeDualSteepestEdgeWeight` : `‖B^{-T} e_p‖²` en espace échelonné,
avec moyenne glissante de `row_ep_density`.
"""
function compute_dual_steepest_edge_weight(e::SimplexEngine, iRow::Int,
    row_ep::HVector)
    clear!(row_ep)
    row_ep.count = 1
    row_ep.index[1] = iRow
    row_ep.array[iRow] = 1.0
    row_ep.packFlag = false
    btran_in_scaled_space!(e.nla, row_ep, e.info.row_ep_density)
    e.info.row_ep_density = update_operation_result_density(
        e.info.row_ep_density, row_ep.count / e.lp.num_row)
    return norm2(row_ep)
end

"""
`HEkk::updateDualSteepestEdgeWeights` :
`w_i += a_i (w_p a_i + Kai y_i)`, plancher `kMinDualSteepestEdgeWeight`.
"""
function update_dual_steepest_edge_weights!(e::SimplexEngine, row_out::Int,
    variable_in::Int, column::HVector, new_pivotal_edge_weight::Float64,
    kai::Float64, dual_steepest_edge_array::AbstractVector{Float64})
    num_row = e.lp.num_row
    col_aq_scale = variable_scale_factor(e.nla, variable_in)
    col_ap_scale = basic_col_scale_factor(e.nla, row_out)
    inv_col_ap_scale = 1.0 / col_ap_scale
    use_row_indices, to_entry = sparse_loop_style(column.count, num_row)
    @inbounds for iEntry ∈ 1:to_entry
        iRow = use_row_indices ? column.index[iEntry] : iEntry
        aa_iRow = column.array[iRow]
        aa_iRow == 0.0 && continue
        dual_steepest_edge_array_value = dual_steepest_edge_array[iRow]
        # `convert_to_scaled_space = !simplex_in_scaled_space_` : la conversion
        # de `HEkk::updateDualSteepestEdgeWeights` est neutre sans facteurs
        # d'échelle, et ignorée quand le LP est échelonné (le facteur y est).
        if !e.lp.is_scaled
            aa_iRow /= basic_col_scale_factor(e.nla, iRow)
            aa_iRow *= col_aq_scale
            dual_steepest_edge_array_value *= inv_col_ap_scale
        end
        e.dual_edge_weight[iRow] +=
            aa_iRow * (new_pivotal_edge_weight * aa_iRow +
                       kai * dual_steepest_edge_array_value)
        e.dual_edge_weight[iRow] = max(kMinDualSteepestEdgeWeight,
            e.dual_edge_weight[iRow])
    end
    return e
end

"""
`HEkk::updateDualDevexWeights` : `w_i = max(w_i, w_p a_i²)` sur les entrées
listées de la colonne pivot.
"""
function update_dual_devex_weights!(e::SimplexEngine, column::HVector,
    new_pivotal_edge_weight::Float64)
    use_row_indices, to_entry = sparse_loop_style(column.count, e.lp.num_row)
    @inbounds for iEntry ∈ 1:to_entry
        iRow = use_row_indices ? column.index[iEntry] : iEntry
        aa_iRow = column.array[iRow]
        e.dual_edge_weight[iRow] = max(e.dual_edge_weight[iRow],
            new_pivotal_edge_weight * aa_iRow * aa_iRow)
    end
    return e
end

"""`HEkk::assessDSEWeightError` — moyennes glissantes d'erreur de poids."""
function assess_dse_weight_error!(e::SimplexEngine,
    computed_edge_weight::Float64, updated_edge_weight::Float64)
    if updated_edge_weight < computed_edge_weight
        relative_deviation = computed_edge_weight / updated_edge_weight
        e.info.average_log_low_DSE_weight_error =
            0.99 * e.info.average_log_low_DSE_weight_error +
            0.01 * log(relative_deviation)
    else
        relative_deviation = updated_edge_weight / computed_edge_weight
        e.info.average_log_high_DSE_weight_error =
            0.99 * e.info.average_log_high_DSE_weight_error +
            0.01 * log(relative_deviation)
    end
    return e
end

"""`HEkk::resetSyntheticClock` — horloge synthétique remise à zéro après INVERT."""
function reset_synthetic_clock!(e::SimplexEngine)
    e.build_synthetic_tick = e.nla.build_synthetic_tick
    e.total_synthetic_tick = 0.0
    return e
end

"""
    rebuild_refactor(e, rebuild_reason)

`HEkk::rebuildRefactor` : avec `no_unnecessary_rebuild_refactor` (défaut), la
ré-inversion n'est faite que si l'erreur du facteur (système test) dépasse
`rebuild_refactor_solution_error_tolerance`.
"""
function rebuild_refactor(e::SimplexEngine, rebuild_reason::Int)
    e.info.update_count == 0 && return false
    refactor = true
    if e.options.no_unnecessary_rebuild_refactor
        if rebuild_reason == kRebuildReasonNo ||
           rebuild_reason == kRebuildReasonPossiblyOptimal ||
           rebuild_reason == kRebuildReasonPossiblyPhase1Feasible ||
           rebuild_reason == kRebuildReasonPossiblyPrimalUnbounded ||
           rebuild_reason == kRebuildReasonPossiblyDualUnbounded ||
           rebuild_reason == kRebuildReasonPrimalInfeasibleInPrimalSimplex
            refactor = false
            error_tolerance = e.options.rebuild_refactor_solution_error_tolerance
            if error_tolerance > 0
                solution_error = factor_solve_error(e)
                refactor = solution_error > error_tolerance
            end
        end
    end
    return refactor
end

"""
    factor_solve_error(e)

`HEkk::factorSolveError` : forme une solution aléatoire à au plus 50 non nuls
(RNG graine 1), résout les systèmes correspondants et mesure l'erreur maximale.
Sert à décider une ré-inversion utile.
"""
function factor_solve_error(e::SimplexEngine)
    num_col = e.lp.num_col
    num_row = e.lp.num_row
    a_matrix = e.lp.a_matrix
    ar_matrix = e.ar_matrix
    basic_index = e.basis.basicIndex
    random = initialise!(e.fse_random, 1)
    ftran_rhs = e.primal_col
    clear!(ftran_rhs)
    solution_num_nz = min(50, (num_row + 1) ÷ 2)
    solution_value = e.fse_solution_value
    solution_index = e.fse_solution_index
    solution_nonzero = e.fse_solution_nonzero
    fill!(solution_nonzero, false)
    sol_count = 0
    @inbounds while true
        iRow = integer(random, num_row) + 1
        solution_nonzero[iRow] && continue
        value = fraction(random)
        sol_count += 1
        solution_value[sol_count] = value
        solution_index[sol_count] = iRow
        solution_nonzero[iRow] = true
        collect_aj!(a_matrix, ftran_rhs, basic_index[iRow], value)
        sol_count == solution_num_nz && break
    end
    # BTRAN : (B^T x) restreint aux colonnes basiques, via la vue rowwise.
    btran_scattered_rhs = e.fse_btran_scattered
    fill!(btran_scattered_rhs, 0.0)
    @inbounds for iX ∈ 1:solution_num_nz
        iRow = solution_index[iX]
        val_X = solution_value[iX]
        for iEl ∈ ar_matrix.p_end[iRow]:(ar_matrix.start[iRow + 1] - 1)
            iCol = ar_matrix.index[iEl]
            btran_scattered_rhs[iCol] +=
                ar_matrix.value[iEl] * val_X
        end
        iCol = num_col + iRow
        if e.basis.nonbasicFlag[iCol] == kNonbasicFlagFalse
            btran_scattered_rhs[iCol] = val_X
        end
    end
    btran_rhs = e.dual_col
    clear!(btran_rhs)
    @inbounds for iRow ∈ 1:num_row
        iCol = basic_index[iRow]
        if btran_scattered_rhs[iCol] != 0.0
            btran_rhs.count += 1
            btran_rhs.array[iRow] = btran_scattered_rhs[iCol]
            btran_rhs.index[btran_rhs.count] = iRow
        end
    end
    expected_density = solution_num_nz * e.info.col_aq_density
    ftran!(e.nla, ftran_rhs, expected_density)
    btran!(e.nla, btran_rhs, expected_density)
    ftran_solution_error = 0.0
    btran_solution_error = 0.0
    @inbounds for iX ∈ 1:solution_num_nz
        iRow = solution_index[iX]
        ftran_solution_error = max(ftran_solution_error,
            abs(ftran_rhs.array[iRow] - solution_value[iX]))
        btran_solution_error = max(btran_solution_error,
            abs(btran_rhs.array[iRow] - solution_value[iX]))
    end
    return max(ftran_solution_error, btran_solution_error)
end

"""
    reinvert_on_numerical_trouble!(e, alpha_col, alpha_row, tol)

`HEkk::reinvertOnNumericalTrouble` : compare les pivots calculés par colonne et
par rangée, relève le seuil de Markowitz si utile, et rend `(ré-inverser,
mesure)`. Rend `false` tant qu'aucune update n'a été faite.
"""
function reinvert_on_numerical_trouble!(e::SimplexEngine,
    alpha_from_col::Float64, alpha_from_row::Float64, tol::Float64)
    abs_col = abs(alpha_from_col)
    abs_row = abs(alpha_from_row)
    min_abs_alpha = min(abs_col, abs_row)
    abs_alpha_diff = abs(abs_col - abs_row)
    numerical_trouble_measure = abs_alpha_diff / min_abs_alpha
    reinvert = numerical_trouble_measure > tol && e.info.update_count > 0
    if reinvert
        current = e.info.factor_pivot_threshold
        new_threshold = 0.0
        if current < kDefaultPivotThreshold
            new_threshold = min(current * kPivotThresholdChangeFactor,
                kDefaultPivotThreshold)
        elseif current < kMaxPivotThreshold && e.info.update_count < 10
            new_threshold = min(current * kPivotThresholdChangeFactor,
                kMaxPivotThreshold)
        end
        if new_threshold != 0.0
            e.info.factor_pivot_threshold = new_threshold
            e.nla.factor.pivot_threshold = new_threshold
        end
    end
    return reinvert, numerical_trouble_measure
end

"""`HEkk::flipBound` — la variable non basique change de borne."""
function flip_bound!(e::SimplexEngine, iCol::Int)
    move = -e.basis.nonbasicMove[iCol]
    e.basis.nonbasicMove[iCol] = move
    e.info.workValue[iCol] = move == 1 ? e.info.workLower[iCol] :
                             e.info.workUpper[iCol]
    return e
end

"""
    update_factor!(e, column, row_ep, iRow, hint)

`HEkk::updateFactor` : mise à jour FT du facteur, puis raisons de ré-inversion
éventuelles (limite d'updates, horloge synthétique). `hint` est la valeur
courante de `rebuild_reason` ; la fonction rend la valeur à jour.
"""
function update_factor!(e::SimplexEngine, column::HVector, row_ep::HVector,
    iRow::Int, hint::Int)
    update!(e.nla, column, row_ep, iRow)
    e.status.has_invert = true
    e.status.has_fresh_invert = false
    if e.info.update_count >= e.info.update_limit
        hint = kRebuildReasonUpdateLimitReached
    end
    reinvert_synthetic_clock = e.total_synthetic_tick >= e.build_synthetic_tick
    performed_min_updates =
        e.info.update_count >= kSyntheticTickReinversionMinUpdateCount
    if reinvert_synthetic_clock && performed_min_updates
        hint = kRebuildReasonSyntheticClockSaysInvert
    end
    return hint
end

"""`HEkk::updatePivots` — entrée/sortie de base, compteurs et drapeaux."""
function update_pivots!(e::SimplexEngine, variable_in::Int, row_out::Int,
    move_out::Int)
    basis = e.basis
    info = e.info
    variable_out = basis.basicIndex[row_out]
    # Hachage de base (détection de cyclage) : sortante puis entrante.
    basis.hash = sparse_inverse_combine(basis.hash, variable_out - 1)
    basis.hash = sparse_combine(basis.hash, variable_in - 1)
    push!(e.visited_basis, basis.hash)
    basis.basicIndex[row_out] = variable_in
    basis.nonbasicFlag[variable_in] = kNonbasicFlagFalse
    basis.nonbasicMove[variable_in] = kNonbasicMoveZe
    info.baseLower[row_out] = info.workLower[variable_in]
    info.baseUpper[row_out] = info.workUpper[variable_in]
    basis.nonbasicFlag[variable_out] = kNonbasicFlagTrue
    if info.workLower[variable_out] == info.workUpper[variable_out]
        info.workValue[variable_out] = info.workLower[variable_out]
        basis.nonbasicMove[variable_out] = kNonbasicMoveZe
    elseif move_out == -1
        info.workValue[variable_out] = info.workLower[variable_out]
        basis.nonbasicMove[variable_out] = kNonbasicMoveUp
    else
        info.workValue[variable_out] = info.workUpper[variable_out]
        basis.nonbasicMove[variable_out] = kNonbasicMoveDn
    end
    info.updated_dual_objective_value +=
        info.workValue[variable_out] * info.workDual[variable_out]
    info.update_count += 1
    if variable_out < e.lp.num_col
        info.num_basic_logicals += 1
    end
    if variable_in < e.lp.num_col
        info.num_basic_logicals -= 1
    end
    e.status.has_invert = false
    e.status.has_fresh_invert = false
    e.status.has_fresh_rebuild = false
    return e
end

"""`HEkk::updateMatrix` — mise à jour de la vue rowwise partitionnée."""
function update_matrix!(e::SimplexEngine, variable_in::Int, variable_out::Int)
    update!(e.ar_matrix, variable_in, variable_out, e.lp.a_matrix)
    return e
end

"""`HEkk::initialisePartitionedRowwiseMatrix` — vue rowwise des non basiques."""
function initialise_partitioned_rowwise_matrix!(e::SimplexEngine)
    e.status.has_ar_matrix && return e
    in_partition = [e.basis.nonbasicFlag[i] == kNonbasicFlagTrue
                    for i ∈ 1:(e.lp.num_col + e.lp.num_row)]
    create_rowwise_partitioned!(e.ar_matrix, e.lp.a_matrix, in_partition)
    e.status.has_ar_matrix = true
    return e
end

"""`HEkk::computeSimplexPrimalInfeasible` — num/max/sum des infaisabilités."""
function compute_simplex_primal_infeasible!(e::SimplexEngine)
    info = e.info
    basis = e.basis
    tolerance = e.options.primal_feasibility_tolerance
    info.num_primal_infeasibilities = 0
    info.max_primal_infeasibility = 0.0
    info.sum_primal_infeasibilities = 0.0
    for i ∈ 1:(e.lp.num_col + e.lp.num_row)
        basis.nonbasicFlag[i] == kNonbasicFlagTrue || continue
        value = info.workValue[i]
        lower = info.workLower[i]
        upper = info.workUpper[i]
        primal_infeasibility = 0.0
        if value < lower - tolerance
            primal_infeasibility = lower - value
        elseif value > upper + tolerance
            primal_infeasibility = value - upper
        end
        if primal_infeasibility > 0
            if primal_infeasibility > tolerance
                info.num_primal_infeasibilities += 1
            end
            info.max_primal_infeasibility =
                max(primal_infeasibility, info.max_primal_infeasibility)
            info.sum_primal_infeasibilities += primal_infeasibility
        end
    end
    for i ∈ 1:e.lp.num_row
        value = info.baseValue[i]
        lower = info.baseLower[i]
        upper = info.baseUpper[i]
        primal_infeasibility = 0.0
        if value < lower - tolerance
            primal_infeasibility = lower - value
        elseif value > upper + tolerance
            primal_infeasibility = value - upper
        end
        if primal_infeasibility > 0
            if primal_infeasibility > tolerance
                info.num_primal_infeasibilities += 1
            end
            info.max_primal_infeasibility =
                max(primal_infeasibility, info.max_primal_infeasibility)
            info.sum_primal_infeasibilities += primal_infeasibility
        end
    end
    return e
end

"""`HEkk::computeSimplexDualInfeasible` — infaisabilités selon `nonbasicMove`."""
function compute_simplex_dual_infeasible!(e::SimplexEngine)
    info = e.info
    basis = e.basis
    tolerance = e.options.dual_feasibility_tolerance
    info.num_dual_infeasibilities = 0
    info.max_dual_infeasibility = 0.0
    info.sum_dual_infeasibilities = 0.0
    for iCol ∈ 1:(e.lp.num_col + e.lp.num_row)
        basis.nonbasicFlag[iCol] == kNonbasicFlagTrue || continue
        dual = info.workDual[iCol]
        lower = info.workLower[iCol]
        upper = info.workUpper[iCol]
        if lower == -kHighsInf && upper == kHighsInf
            dual_infeasibility = abs(dual)
        else
            dual_infeasibility = -basis.nonbasicMove[iCol] * dual
        end
        if dual_infeasibility > 0
            if dual_infeasibility >= tolerance
                info.num_dual_infeasibilities += 1
            end
            info.max_dual_infeasibility =
                max(dual_infeasibility, info.max_dual_infeasibility)
            info.sum_dual_infeasibilities += dual_infeasibility
        end
    end
    return e
end

"""`HEkk::computeSimplexInfeasible`."""
function compute_simplex_infeasible!(e::SimplexEngine)
    compute_simplex_primal_infeasible!(e)
    compute_simplex_dual_infeasible!(e)
    return e
end

"""
    return_from_solve!(e, algorithm)

`HEkk::returnFromSolve` : état normalisé rendu par un solve. Retire shifts et
perturbations, recalcule valeurs primales, duals et infaisabilités selon le
statut, remet `valid_backtracking_basis` à faux et annule les duals des
basiques. C'est cet état que `getSolution` rapporte, et il est nécessaire avant
que le primal ne classe un `kUnboundedOrInfeasible` (le dual sort de sa phase 1
avec des bornes spéciales). Les statuts d'erreur et les rapports ne sont pas
portés.
"""
function return_from_solve!(e::SimplexEngine, algorithm::SimplexAlgorithm)
    status = e.model_status
    info = e.info
    info.valid_backtracking_basis = false
    status == kSolveError && return e
    if status != kOptimal
        invalidate_primal_infeasibility_record!(e)
        invalidate_dual_infeasibility_record!(e)
    end
    if status == kInfeasible
        if algorithm == kPrimal
            # Après une infaisabilité prouvée en phase 1 primale, les duals
            # sont recalculés avec les coûts du LP.
            initialise_cost!(e, kDual, kSolvePhase2)
            compute_dual!(e)
        end
        compute_simplex_infeasible!(e)
    elseif status == kUnboundedOrInfeasible
        # Bornes du LP, primals et infaisabilités recalculés : le primal
        # tranchera sur cet état.
        initialise_bound!(e, kDual, kSolvePhase2)
        compute_primal!(e)
        compute_simplex_infeasible!(e)
    elseif status == kUnbounded
        compute_simplex_infeasible!(e)
    elseif status != kOptimal
        # Limite atteinte (itérations, temps, objectif) ou statut inconnu :
        # bornes et coûts du LP, valeurs et duals recalculés.
        initialise_bound!(e, kDual, kSolvePhase2)
        initialise_nonbasic_value_and_move!(e)
        compute_primal!(e)
        initialise_cost!(e, kDual, kSolvePhase2)
        compute_dual!(e)
        compute_simplex_infeasible!(e)
    end
    for iRow ∈ 1:e.lp.num_row
        info.workDual[e.basis.basicIndex[iRow]] = 0.0
    end
    compute_primal_objective_value!(e)
    # `HApp::solveLpSimplex` : la base rendue est mémorisée (warm start du
    # prochain solve si la base du Ekk a été jetée entre-temps).
    store_solution_basis!(e)
    return e
end

"""
`HEkk::computeSimplexLpDualInfeasible` — infaisabilités duales selon les bornes
du LP (utilisée pour conclure en phase 1). Rend `(num, max, sum)`.
"""
function compute_simplex_lp_dual_infeasible(e::SimplexEngine)
    info = e.info
    basis = e.basis
    tolerance = e.options.dual_feasibility_tolerance
    num = 0
    max_infeasibility = 0.0
    sum_infeasibility = 0.0
    for iCol ∈ 1:e.lp.num_col
        basis.nonbasicFlag[iCol] == kNonbasicFlagTrue || continue
        dual = info.workDual[iCol]
        lower = e.lp.col_lower[iCol]
        upper = e.lp.col_upper[iCol]
        if upper == kHighsInf
            dual_infeasibility = lower == -kHighsInf ? abs(dual) : -dual
        else
            dual_infeasibility = lower == -kHighsInf ? dual : 0.0
        end
        if dual_infeasibility > 0
            if dual_infeasibility >= tolerance
                num += 1
            end
            max_infeasibility = max(dual_infeasibility, max_infeasibility)
            sum_infeasibility += dual_infeasibility
        end
    end
    for iRow ∈ 1:e.lp.num_row
        iVar = e.lp.num_col + iRow
        basis.nonbasicFlag[iVar] == kNonbasicFlagTrue || continue
        dual = -info.workDual[iVar]
        lower = e.lp.row_lower[iRow]
        upper = e.lp.row_upper[iRow]
        if upper == kHighsInf
            dual_infeasibility = lower == -kHighsInf ? abs(dual) : -dual
        else
            dual_infeasibility = lower == -kHighsInf ? dual : 0.0
        end
        if dual_infeasibility > 0
            if dual_infeasibility >= tolerance
                num += 1
            end
            max_infeasibility = max(dual_infeasibility, max_infeasibility)
            sum_infeasibility += dual_infeasibility
        end
    end
    return num, max_infeasibility, sum_infeasibility
end

"""`HEkk::invalidatePrimalInfeasibilityRecord`."""
function invalidate_primal_infeasibility_record!(e::SimplexEngine)
    e.info.num_primal_infeasibilities = kHighsIllegalInfeasibilityCount
    e.info.max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure
    e.info.sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure
    return e
end

"""`HEkk::invalidatePrimalMaxSumInfeasibilityRecord`."""
function invalidate_primal_max_sum_infeasibility_record!(e::SimplexEngine)
    e.info.max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure
    e.info.sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure
    return e
end

"""
    apply_taboo_variable_in!(e, values, overwrite_with)

`HEkk::applyTabooVariableIn` : masque les valeurs des variables entrantes
taboues (le dual les verrait comme non attractives le temps d'un CHUZC).
`values` est typiquement `workDual`.
"""
function apply_taboo_variable_in!(e::SimplexEngine, values::Vector{Float64},
    overwrite_with::Float64)
    for change ∈ e.bad_basis_change
        if change.taboo
            change.save_value = values[change.variable_in]
            values[change.variable_in] = overwrite_with
        end
    end
    return e
end

"""`HEkk::unapplyTabooVariableIn` — parcourt en ordre inverse (voir source)."""
function unapply_taboo_variable_in!(e::SimplexEngine,
    values::Vector{Float64})
    for iX ∈ length(e.bad_basis_change):-1:1
        change = e.bad_basis_change[iX]
        if change.taboo
            values[change.variable_in] = change.save_value
        end
    end
    return e
end

"""`HEkk::invalidateDualInfeasibilityRecord`."""
function invalidate_dual_infeasibility_record!(e::SimplexEngine)
    e.info.num_dual_infeasibilities = kHighsIllegalInfeasibilityCount
    e.info.max_dual_infeasibility = kHighsIllegalInfeasibilityMeasure
    e.info.sum_dual_infeasibilities = kHighsIllegalInfeasibilityMeasure
    return e
end

"""
    bailout!(e)

`HEkk::bailout` (sous-ensemble : limite d'itérations et de temps ; les
callbacks et la limite d'objectif ne sont pas portés). Pose `solve_bailout` et
le statut du modèle quand une limite est atteinte.
"""
function bailout!(e::SimplexEngine)
    e.solve_bailout && return true
    if e.options.time_limit < kHighsInf &&
       (time() - e.solve_start_time) > e.options.time_limit
        e.solve_bailout = true
        e.model_status = kTimeLimit
    elseif e.iteration_count - e.iteration_count0 >=
           e.options.simplex_iteration_limit
        e.solve_bailout = true
        e.model_status = kIterationLimit
    end
    return e.solve_bailout
end

"""`HEkk::choosePriceTechnique` — prix colonne ou rowwise avec bascule."""
function choose_price_technique(e::SimplexEngine, row_ep_density::Float64)
    density_for_column_price_switch = 0.75
    price_strategy = e.info.price_strategy
    use_col_price = price_strategy == kSimplexPriceStrategyCol ||
                    (price_strategy == kSimplexPriceStrategyRowSwitchColSwitch &&
                     row_ep_density > density_for_column_price_switch)
    use_row_price_w_switch =
        price_strategy == kSimplexPriceStrategyRowSwitchColSwitch
    return use_col_price, use_row_price_w_switch
end

"""
    tableau_row_price!(e, row_ep, row_ap, quad_precision = false)

`HEkk::tableauRowPrice` : `row_ap = row_ep' A` sur les non basiques, par
`priceByColumn` ou `priceByRowWithSwitch` (vue partitionnée), puis moyenne
glissante de la densité de `row_ap`. `quad_precision` (utilisé par
`improveChooseColumnRow`) accumule en double-double et exige que `row_ep`
courant soit à l'échelle du facteur.
"""
function tableau_row_price!(e::SimplexEngine, row_ep::HVector, row_ap::HVector,
    quad_precision::Bool=false)
    solver_num_row = e.lp.num_row
    local_density = row_ep.count / solver_num_row
    use_col_price, use_row_price_w_switch = choose_price_technique(e,
        local_density)
    clear!(row_ap)
    if use_col_price
        price_by_column!(e.lp.a_matrix, row_ap, row_ep, quad_precision)
        for iCol ∈ 1:e.lp.num_col
            row_ap.array[iCol] *= e.basis.nonbasicFlag[iCol]
        end
    elseif use_row_price_w_switch
        price_by_row_with_switch!(e.ar_matrix, row_ap, row_ep,
            e.info.row_ap_density, 1, kHyperPriceDensity, quad_precision)
    else
        price_by_row!(e.ar_matrix, row_ap, row_ep, quad_precision)
    end
    e.info.row_ap_density = update_operation_result_density(
        e.info.row_ap_density, row_ap.count / e.lp.num_col)
    return row_ap
end

"""
    unit_btran_residual!(e, row_out, row_ep, residual)

`HEkk::unitBtranResidual` : `residual = e_row_out - Bᵀ row_ep`, accumulé en
double-double (le résidu est une différence de quantités proches ; en double
le bruit d'arrondi dominerait). Rend la norme infinie du résidu.
"""
function unit_btran_residual!(e::SimplexEngine, row_out::Int, row_ep::HVector,
    residual::HVector)
    lp = e.lp
    quad_residual = zeros(CDouble, lp.num_row)
    quad_residual[row_out] = CDouble(-1.0)
    for iRow ∈ 1:lp.num_row
        iVar = e.basis.basicIndex[iRow]
        value = quad_residual[iRow]
        if iVar <= lp.num_col
            for iEl ∈ lp.a_matrix.start[iVar]:(lp.a_matrix.start[iVar + 1] - 1)
                value += lp.a_matrix.value[iEl] *
                         row_ep.array[lp.a_matrix.index[iEl]]
            end
        else
            value += row_ep.array[iVar - lp.num_col]
        end
        quad_residual[iRow] = value
    end
    clear!(residual)
    residual.packFlag = false
    residual_norm = 0.0
    for iRow ∈ 1:lp.num_row
        value = Float64(quad_residual[iRow])
        if value != 0.0
            residual.array[iRow] = value
            residual.count += 1
            residual.index[residual.count] = iRow
        end
        residual_norm = max(abs(residual.array[iRow]), residual_norm)
    end
    return residual_norm
end

"""
    unit_btran_iterative_refinement!(e, row_out, row_ep)

`HEkk::unitBtranIterativeRefinement` : une passe de raffinement itératif du
BTRAN unitaire `row_ep`. Le résidu est normalisé par la puissance de deux la
plus proche avant le BTRAN (pour que `kHighsTiny` ne s'applique pas à tort à
un résidu minuscule), puis la correction est retirée et la liste d'indices est
reconstruite dans l'ordre croissant des lignes.
"""
function unit_btran_iterative_refinement!(e::SimplexEngine, row_out::Int,
    row_ep::HVector)
    residual = HVector(e.lp.num_row)
    residual_norm = unit_btran_residual!(e, row_out, row_ep, residual)
    residual_norm == 0.0 && return e
    residual_scale = nearest_power_of_two_scale(residual_norm)
    for iEl ∈ 1:residual.count
        residual.array[residual.index[iEl]] *= residual_scale
    end
    btran!(e.nla, residual, 1.0)
    row_ep.count = 0
    for iRow ∈ 1:e.lp.num_row
        if residual.array[iRow] != 0.0
            correction_value = residual.array[iRow] / residual_scale
            row_ep.array[iRow] -= correction_value
        end
        if abs(row_ep.array[iRow]) < kHighsTiny
            row_ep.array[iRow] = 0.0
        else
            row_ep.count += 1
            row_ep.index[row_ep.count] = iRow
        end
    end
    return e
end

"""`nearestPowerOfTwoScale` (`HighsUtils.cpp:1211`)."""
function nearest_power_of_two_scale(value::Float64)
    check_x, exp_scale = frexp(value)
    if abs(check_x) == 0.5
        check_x *= 2
        exp_scale -= 1
    end
    return ldexp(1.0, -exp_scale)
end

"""`HEkk::getValueScale` — échelle du pivot (puissance de deux la plus proche)."""
function get_value_scale(count::Int, value::AbstractVector{Float64})
    count <= 0 && return 1.0
    max_abs_value = 0.0
    for iX ∈ 1:count
        max_abs_value = max(abs(value[iX]), max_abs_value)
    end
    return nearest_power_of_two_scale(max_abs_value)
end

"""`HEkk::getMaxAbsRowValue` (vue rowwise initialisée si besoin)."""
function get_max_abs_row_value(e::SimplexEngine, row::Int)
    initialise_partitioned_rowwise_matrix!(e)
    val = -1.0
    for i ∈ e.ar_matrix.start[row]:(e.ar_matrix.start[row + 1] - 1)
        val = max(val, abs(e.ar_matrix.value[i]))
    end
    return val
end

"""
    proof_of_primal_infeasibility!(e, row_ep, move_out, row_out)

`HEkk::proofOfPrimalInfeasibility` : cherche une combinaison `y = row_ep' A`
dont la borne supérieure impliquée contredit la borne inférieure de la
contrainte. Les accumulations `proof_lower`/`implied_upper`/`sumInf` sont en
`CDouble` (double-double, comme la source) : en `Float64`, le bruit
d'arrondi rendait le gap positif et produisait de fausses preuves
d'infaisabilité sur les séquences échelonnées.
"""
function proof_of_primal_infeasibility!(e::SimplexEngine, row_ep::HVector,
    move_out::Int, row_out::Int)
    lp = e.lp
    proof_lower = CDouble(0.0)
    # Raffine `row_ep` : contributions négligeables et bornes infinies.
    for iX ∈ 1:row_ep.count
        iRow = row_ep.index[iX]
        row_ep_value = row_ep.array[iRow]
        row_ep_value == 0.0 && continue
        if abs(row_ep_value * get_max_abs_row_value(e, iRow)) <=
           e.options.small_matrix_value
            row_ep.array[iRow] = 0.0
            continue
        end
        row_ep.array[iRow] *= move_out
        if row_ep.array[iRow] > 0
            rowBound = lp.row_lower[iRow]
            if rowBound == -kHighsInf
                row_ep.array[iRow] = 0.0
                continue
            end
        else
            rowBound = lp.row_upper[iRow]
            if rowBound == kHighsInf
                row_ep.array[iRow] = 0.0
                continue
            end
        end
        proof_lower += row_ep.array[iRow] * rowBound
    end
    # Coefficients de la preuve : `row_ep' ar_matrix`, accumulation en double.
    proof_scattered = zeros(lp.num_col)
    for iRow ∈ 1:lp.num_row
        multiplier = row_ep.array[iRow]
        multiplier == 0.0 && continue
        for iEl ∈ e.ar_matrix.start[iRow]:(e.ar_matrix.start[iRow + 1] - 1)
            iCol = e.ar_matrix.index[iEl]
            proof_scattered[iCol] += multiplier * e.ar_matrix.value[iEl]
        end
    end
    proof_value = Float64[]
    proof_index = Int[]
    for iCol ∈ 1:lp.num_col
        value = proof_scattered[iCol]
        if abs(value) > kHighsTiny
            push!(proof_value, value)
            push!(proof_index, iCol)
        end
    end
    implied_upper = CDouble(0.0)
    sum_inf = CDouble(0.0)
    for i ∈ eachindex(proof_value)
        iCol = proof_index[i]
        value = proof_value[i]
        if value > 0
            if lp.col_upper[iCol] == kHighsInf
                sum_inf += value
                sum_inf > e.options.small_matrix_value && break
                continue
            end
            implied_upper += value * lp.col_upper[iCol]
        else
            if lp.col_lower[iCol] == -kHighsInf
                sum_inf += -value
                sum_inf > e.options.small_matrix_value && break
                continue
            end
            implied_upper += value * lp.col_lower[iCol]
        end
    end
    infinite_implied_upper = sum_inf > e.options.small_matrix_value
    gap = Float64(proof_lower - implied_upper)
    gap_ok = gap > e.options.primal_feasibility_tolerance
    return !infinite_implied_upper && gap_ok
end
