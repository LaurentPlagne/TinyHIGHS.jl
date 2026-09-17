# Portage de `highs/util/HFactor.{h,cpp}` + `HFactorRefactor`/`HFactorUtils`
# (licence MIT, HiGHS). Périmètre M1a : `build` (buildSimple → buildKernel →
# buildFinish) et les solves FTRAN/BTRAN (sparses et hyper-sparses), méthode
# d'update FT (les buffers d'update sont vides en M1a).
#
# Convention 1-based (plan §2.1) : les tableaux de structure stockent des
# **positions** (déjà décalées) ; les valeurs encodées (`mr_count_before`
# négatif, marqueurs de listes `-2 - count`) gardent l'encodage de la source.
# `permute` : 0 = non pivoté (sentinelle `-1` de la source).
#
# Non porté en M1a : carence de rang complète (`buildHandleRankDeficiency`,
# `buildMarkSingC`), mises à jour (`updateFT`, PF/MPF/APF), `rebuild`, add/delete
# colonnes/lignes. `build!` rend la carence de rang sans compléter le facteur.

"""
    RefactorInfo

Information de refactorisation (`RefactorInfo`, HStruct.h:43) : suite des pivots
du dernier `build` réussi, rejouable par `rebuild!` sans nouvelle recherche de
Markowitz. `pivot_row`/`pivot_var` sont 1-based ; `use` est armé par l'appelant
(le simplexe), jamais par `build`.
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

Factorisation de la matrice de base : `P B Q = L U` (cf. `HFactor` de HiGHS).
`a_start/a_index/a_value` forment la matrice des contraintes en CSC 1-based ;
`basic_index` porte les numéros de variable 1-based (colonnes `1:num_col`,
logiques `num_col+1:num_col+num_row`).
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
        throw(ArgumentError("dimensions négatives"))
    (length(a_start) == num_col + 1 && length(a_index) == length(a_value) &&
     length(basic_index) == num_basic) ||
        throw(ArgumentError("tailles de a_start/a_index/a_value/basic_index incohérentes"))
    b_max_dim = max(num_row, num_basic)
    # `basis_matrix_limit_size` : borne du nombre d'entrées de B (HFactor.cpp:251)
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
    # `b_start[1] = 1` : premier emplacement de la première colonne (la source
    # laisse `b_start[0] = 0`, l'offset 0-based).
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
        zeros(Int, num_row), zeros(Int, 0), zeros(0), zeros(Int, 0), zeros(Int, 0),
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
    build!(f)

Forme `P B Q = L U`. Rend `0` si le facteur est complet, sinon la carence de
rang résiduelle. Si `refactor_info.use` est armé, passe par `rebuild!` ; une
base déficiente est complétée par des logiques (`buildHandleRankDeficiency!` +
`buildMarkSingC!`) comme dans la source.
"""
function build!(f::HFactor)
    # Refactorisation depuis la liste de pivots du dernier build réussi.
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
    # `HFactor.cpp:453` : `l_start[num_row] + u_last_p[num_row-1] + num_row`,
    # soit le nombre d'entrées de L, de U, plus les pivots. En 1-based,
    # `l_start[num_row+1] - 1` et `u_last_p[num_row] - 1` sont les comptes
    # (u_last_p est une fin exclusive) ; pour `num_row == 0` la source lit
    # `u_last_p[-1]` (hors bornes) : le port définit la valeur à 0.
    if f.num_row > 0
        f.invert_num_el = (f.l_start[f.num_row + 1] - 1) +
                          (f.u_last_p[f.num_row] - 1) + f.num_row
    else
        f.invert_num_el = 0
    end
    f.kernel_dim -= f.rank_deficiency
    return f.rank_deficiency
end
