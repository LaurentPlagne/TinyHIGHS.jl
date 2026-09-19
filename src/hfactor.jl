# Port of `highs/util/HFactor.{h,cpp}` + `HFactorRefactor`/`HFactorUtils`
# (MIT License, HiGHS). Scope: `build` (buildSimple -> buildKernel ->
# buildFinish) and sparse / hyper-sparse FTRAN/BTRAN solves, Forrest-Tomlin update method.
#
# 1-based indexing: structural arrays store 1-based positions;
# encoded values (`mr_count_before` negative, list marker `-2 - count`)
# retain upstream's exact encoding. `permute`: 0 = unpivoted (upstream sentinel `-1`).

"""
    RefactorInfo

Refactorization information (`RefactorInfo`, HStruct.h:43): sequence of pivots
from the last successful `build`, replayable by `rebuild!` without repeating Markowitz search.
`pivot_row`/`pivot_var` are 1-based; `use` is activated by the caller (simplex engine),
never by `build`.
"""
mutable struct RefactorInfo
    use::Bool
    pivot_row::Vector{Int}
    pivot_var::Vector{Int}
    pivot_type::Vector{Int8}
    build_synthetic_tick::Float64
end

RefactorInfo() = RefactorInfo(false, Int[], Int[], Int8[], 0.0)

function clear!(info::RefactorInfo)
    info.use = false
    info.build_synthetic_tick = 0.0
    empty!(info.pivot_var)
    empty!(info.pivot_row)
    empty!(info.pivot_type)
    return info
end

"""
    HFactor(num_col, num_row, num_basic, a_start, a_index, a_value, basic_index;
            pivot_threshold, pivot_tolerance, update_method)

Sparse LU factorization and Forrest-Tomlin basis update engine, equivalent to HiGHS's `HFactor`.

Computes and maintains the factorization:
```math
P B Q = L U
```
where \$B\$ is the basis matrix composed of columns indexed by `basic_index`, \$P\$ and \$Q\$
are row and column permutation matrices, \$L\$ is unit lower triangular, and \$U\$ is upper
triangular.

# Key Optimizations
- **Branchless reciprocal substitution**: Pre-inverts diagonal pivots and replaces
  per-pivot divisions in `ftranU`, `btranU`, and `solveHyper` with multiplication.
- **Hyper-sparse forward and backward transformation**: Graph reachability search using
  `HVector` indices for sub-linear FTRAN / BTRAN runtime.
- **Forrest-Tomlin updates**: In-place update of factorization across basis changes.
"""
mutable struct HFactor
    num_row::Int
    num_col::Int
    num_basic::Int
    inv_num_row::Float64
    a_start::Vector{Int}
    a_index::Vector{Int}
    a_value::Vector{Float64}
    basic_index::Vector{Int}
    pivot_threshold::Float64
    pivot_tolerance::Float64
    update_method::Int

    basis_matrix_limit_size::Int
    basis_matrix_num_el::Int
    invert_num_el::Int
    kernel_dim::Int
    kernel_num_el::Int
    build_synthetic_tick::Float64
    rank_deficiency::Int
    nwork::Int

    b_var::Vector{Int}
    b_start::Vector{Int}
    b_index::Vector{Int}
    b_value::Vector{Float64}
    permute::Vector{Int}

    mc_var::Vector{Int}
    mc_start::Vector{Int}
    mc_count_a::Vector{Int}
    mc_count_n::Vector{Int}
    mc_space::Vector{Int}
    mc_index::Vector{Int}
    mc_value::Vector{Float64}
    mc_min_pivot::Vector{Float64}

    mr_start::Vector{Int}
    mr_count::Vector{Int}
    mr_space::Vector{Int}
    mr_count_before::Vector{Int}
    mr_index::Vector{Int}

    mwz_column_index::Vector{Int}
    mwz_column_mark::Vector{Bool}
    mwz_column_array::Vector{Float64}

    col_link_first::Vector{Int}
    col_link_next::Vector{Int}
    col_link_last::Vector{Int}
    row_link_first::Vector{Int}
    row_link_next::Vector{Int}
    row_link_last::Vector{Int}

    l_pivot_lookup::Vector{Int}
    l_pivot_index::Vector{Int}
    l_start::Vector{Int}
    l_index::Vector{Int}
    l_value::Vector{Float64}
    lr_start::Vector{Int}
    lr_index::Vector{Int}
    lr_value::Vector{Float64}

    u_pivot_lookup::Vector{Int}
    u_pivot_index::Vector{Int}
    u_pivot_value::Vector{Float64}
    u_pivot_inv_value::Vector{Float64}
    u_start::Vector{Int}
    u_last_p::Vector{Int}
    u_index::Vector{Int}
    u_value::Vector{Float64}
    u_merit_x::Int
    u_total_x::Int
    ur_start::Vector{Int}
    ur_lastp::Vector{Int}
    ur_space::Vector{Int}
    ur_index::Vector{Int}
    ur_value::Vector{Float64}

    pf_pivot_value::Vector{Float64}
    pf_pivot_index::Vector{Int}
    pf_start::Vector{Int}
    pf_index::Vector{Int}
    pf_value::Vector{Float64}

    refactor_info::RefactorInfo
    row_with_no_pivot::Vector{Int}
    col_with_no_pivot::Vector{Int}
    var_with_no_pivot::Vector{Int}

    iwork::Vector{Int}
    dwork::Vector{Float64}
end

function HFactor(num_col::Int, num_row::Int, num_basic::Int,
    a_start::Vector{Int}, a_index::Vector{Int}, a_value::Vector{Float64},
    basic_index::Vector{Int};
    pivot_threshold::Float64=kDefaultPivotThreshold,
    pivot_tolerance::Float64=kDefaultPivotTolerance,
    update_method::Int=kUpdateMethodFt)
    (num_col >= 0 && num_row >= 0 && num_basic >= 0) ||
        throw(ArgumentError("dimensions must be non-negative"))
    (length(a_start) == num_col + 1 && length(a_index) == length(a_value) &&
     length(basic_index) == num_basic) ||
        throw(ArgumentError("inconsistent dimensions in a_start/a_index/a_value/basic_index"))
    b_max_dim = max(num_row, num_basic)
    # basis_matrix_limit_size: upper bound on number of nonzeros in B (HFactor.cpp:251)
    counts = zeros(Int, num_row + 1)
    for i in 1:num_col
        counts[a_start[i + 1] - a_start[i] + 1] += 1
    end
    limit = 0
    counted = 0
    for i in num_row:-1:0
        counted >= b_max_dim && break
        limit += i * counts[i + 1]
        counted += counts[i + 1]
    end
    limit += b_max_dim
    th = clamp(pivot_threshold, kMinPivotThreshold, kMaxPivotThreshold)
    tol = clamp(pivot_tolerance, kMinPivotTolerance, kMaxPivotTolerance)
    # b_start[1] = 1: 1-based start offset of the first column (upstream b_start[0] = 0).
    b_start = zeros(Int, b_max_dim + 1)
    b_start[1] = 1

    return HFactor(num_row, num_col, num_basic, 1.0 / num_row,
        a_start, a_index, a_value, copy(basic_index), th, tol, update_method,
        limit, 0, 0, 0, 0, 0.0, 0, 0,
        zeros(Int, b_max_dim), b_start,
        zeros(Int, limit), zeros(limit), zeros(Int, b_max_dim),
        zeros(Int, num_basic), zeros(Int, num_basic), zeros(Int, num_basic),
        zeros(Int, num_basic), zeros(Int, num_basic),
        zeros(Int, limit * kMCExtraEntriesMultiplier),
        zeros(limit * kMCExtraEntriesMultiplier), zeros(num_basic),
        zeros(Int, num_row), zeros(Int, num_row), zeros(Int, num_row),
        zeros(Int, num_row), zeros(Int, limit * kMRExtraEntriesMultiplier),
        zeros(Int, num_row), zeros(Bool, num_row), zeros(num_row),
        fill(-1, num_row + 1), zeros(Int, num_basic), zeros(Int, num_basic),
        fill(-1, num_basic + 1), zeros(Int, num_row), zeros(Int, num_row),
        zeros(Int, num_row), zeros(Int, num_row), [1], zeros(Int, 0), zeros(0),
        zeros(Int, num_row + 1), zeros(Int, 0), zeros(0),
        zeros(Int, num_row), zeros(Int, 0), zeros(0), zeros(0), zeros(Int, 0), zeros(Int, 0),
        zeros(Int, 0), zeros(0), 0, 0,
        zeros(Int, num_row + 1), zeros(Int, 0), zeros(Int, 0), zeros(Int, 0),
        zeros(0),
        zeros(0), zeros(Int, 0), [1], zeros(Int, 0), zeros(0),
        RefactorInfo(), zeros(Int, 0), zeros(Int, 0), zeros(Int, 0),
        zeros(Int, num_basic + 1), zeros(num_row))
end

include("hfactor_utils.jl")
include("hfactor_build.jl")
include("hfactor_solve.jl")
include("hfactor_refactor.jl")

"""
    build!(f::HFactor) -> Int

Compute the sparse LU factorization \$P B Q = L U\$ for the current basis matrix.

Returns `0` if the basis is non-singular and the factorization completed successfully,
or the rank deficiency count if the basis is rank-deficient (in which case singular columns
are replaced by logical slacks).
"""
function build!(f::HFactor)
    # Refactorization using pivot sequence from last successful build.
    if f.refactor_info.use
        rank_deficiency = rebuild!(f)
        rank_deficiency == 0 && return 0
    end
    clear!(f.refactor_info)
    f.build_synthetic_tick = 0.0
    buildSimple!(f)
    f.rank_deficiency = buildKernel!(f)
    incomplete_basis = f.num_basic < f.num_row
    if f.rank_deficiency != 0 || incomplete_basis
        buildHandleRankDeficiency!(f)
        buildMarkSingC!(f)
    end
    if incomplete_basis
        clear!(f.refactor_info)
        return f.rank_deficiency - (f.num_row - f.num_basic)
    end
    buildFinish!(f)
    if f.rank_deficiency != 0
        clear!(f.refactor_info)
    else
        f.refactor_info.build_synthetic_tick = f.build_synthetic_tick
    end
    # `HFactor.cpp:453`: `l_start[num_row] + u_last_p[num_row-1] + num_row`,
    # i.e., nonzeros in L and U plus diagonal pivots. In 1-based indexing,
    # `l_start[num_row+1] - 1` and `u_last_p[num_row] - 1` are the element counts
    # (u_last_p is exclusive end); for `num_row == 0` upstream reads `u_last_p[-1]`:
    # ported version safely guards to 0.
    if f.num_row > 0
        f.invert_num_el = (f.l_start[f.num_row + 1] - 1) +
                          (f.u_last_p[f.num_row] - 1) + f.num_row
    else
        f.invert_num_el = 0
    end
    f.kernel_dim -= f.rank_deficiency
    return f.rank_deficiency
end
