"""
    TinyHiGHS

Portage Julia pur et autonome du simplexe révisé dual et primal de HiGHS (MIT).
Optimisé pour la résolution haute performance de problèmes d'optimisation linéaire
(LPs) récurrents, de flots sur réseaux et de chaînes de sous-problèmes sans allocation.
"""
module TinyHiGHS

export HVector, SimplexBasis, SimplexInfo, HFactor, SparseMatrix, setup!, clear!,
    clearScalars!, tight!, pack!, reIndex!, norm2, saxpy!, build!, ftranCall!,
    btranCall!, update!, rebuild!, kHighsTiny, kHighsZero,
    is_colwise, is_rowwise, num_nz, ensure_colwise!, ensure_rowwise!, get_row,
    get_col, product!, product_transpose!, alpha_product_plus_y!, compute_dot,
    collect_aj!, price_by_column!, price_by_row!, price_by_row_with_switch!,
    create_rowwise_partitioned!, Scale, scale_col!, scale_row!, apply_scale!,
    apply_col_scale!, apply_row_scale!, unapply_scale!, Nla, invert!, ftran!,
    btran!, ftran_in_scaled_space!, btran_in_scaled_space!,
    apply_basis_matrix_row_scale!, apply_basis_matrix_col_scale!,
    unapply_basis_matrix_row_scale!, variable_scale_factor,
    basic_col_scale_factor, pivot_in_scaled_space, row_ep_2norm_in_scaled_space,
    transform_for_update!, SimplexLp, SimplexOptions, SimplexEngine, ObjSense,
    SimplexAlgorithm, kMinimize, kMaximize, kPrimal, kDual, kSolvePhaseUnknown,
    kSolvePhase1, kSolvePhase2, kNonbasicFlagTrue, kNonbasicFlagFalse,
    kNonbasicMoveUp, kNonbasicMoveDn, kNonbasicMoveZe, kIllegalMoveValue,
    kHighsInf, kHighsIllegalInfeasibilityCount,
    kHighsIllegalInfeasibilityMeasure, set_basis!, set_nonbasic_move!,
    initialise_bound!, initialise_cost!, initialise_lp_col_bound!,
    initialise_lp_row_bound!, initialise_lp_col_cost!, initialise_lp_row_cost!,
    initialise_nonbasic_value_and_move!, compute_primal!, compute_dual!,
    compute_primal_objective_value!, compute_dual_objective_value!,
    HighsRandom, ModelStatus, SimplexStatus, DualSolver, DualRHS, DualRow,
    initialise_for_solve!, solve!, reset!, sparse_combine, sparse_inverse_combine,
    basis_hash, BadBasisChange, add_bad_basis_change!,
    clear_bad_basis_change!, clear_bad_basis_change_taboo_flag!,
    taboo_bad_basis_change, apply_taboo_row_out!, unapply_taboo_row_out!,
    update_bad_basis_change!, is_bad_basis_change!,
    get_nonsingular_inverse!, put_backtracking_basis!,
    get_backtracking_basis!, kBadBasisChangeAll, kBadBasisChangeSingular,
    kBadBasisChangeCycling, kBadBasisChangeFailedInfeasibilityProof,
    PrimalSolver, apply_taboo_variable_in!, unapply_taboo_variable_in!,
    update_status!, change_col_bounds!, change_cols_bounds!,
    change_row_bounds!, change_rows_bounds!, change_cols_cost!,
    change_objective_sense!, change_coeff!, kLpActionScale, kLpActionNewCosts,
    kLpActionNewBounds, kLpActionNewRows, consider_scaling!, scale_lp!,
    apply_lp_scale!, unapply_lp_scale!, clear_scale!, clear_scaling!,
    unscale_simplex!, restore_scale!, kSimplexScaleStrategyOff,
    kSimplexScaleStrategyChoose, kSimplexScaleStrategyEquilibration,
    kSimplexScaleStrategyForcedEquilibration, kSimplexScaleStrategyMaxValue,
    read_lp, write_lp, solve_lp

include("constants.jl")
include("hvector.jl")
include("random.jl")
include("types.jl")
include("c_double.jl")
include("hfactor.jl")
include("sparse_matrix.jl")
include("nla.jl")
include("engine.jl")
include("scaling.jl")
include("dual_rhs.jl")
include("dual_row.jl")
include("dual.jl")
include("primal.jl")
include("lp_io.jl")

end
