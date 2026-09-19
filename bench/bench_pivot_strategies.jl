# ==============================================================================
# Micro-benchmark comparatif des stratégies de pivot dans HFactor :
#   1. kPivotBranching  : court-circuit conditionnel (if pivot == ±1.0)
#   2. kPivotBranchless : multiplication branchless par pré-inverse SIMD (* inv_pivot)
#   3. kPivotFdiv       : division inconditionnelle (/ pivot, HiGHS C++ d'origine)
# ==============================================================================

using Printf
using Random
using TinyHiGHS

"""
Construit un HFactor synthétique avec une matrice triangulaire supérieure creuse
et une distribution contrôlée des pivots.
"""
function make_synthetic_hfactor(n::Int, pivot_mode::Symbol)
    # Matrice identité initiale comme base
    a_start = collect(1:(n + 1))
    a_index = collect(1:n)
    a_value = ones(Float64, n)
    basic_index = collect(1:n)

    f = HFactor(n, n, n, a_start, a_index, a_value, basic_index)
    build!(f)

    # Peupler u_pivot_value selon le profil souhaité
    rng = Xoshiro(42)
    if pivot_mode == :network
        # 90% de 1.0, 5% de -1.0, 5% de valeurs arbitraires
        for i in 1:n
            r = rand(rng)
            f.u_pivot_value[i] = r < 0.90 ? 1.0 : (r < 0.95 ? -1.0 : 0.5 + rand(rng))
        end
    elseif pivot_mode == :general
        # 100% de réels arbitraires continus (aucun 1.0)
        for i in 1:n
            f.u_pivot_value[i] = 0.1 + 2.0 * rand(rng)
        end
    elseif pivot_mode == :mixed_mispredict
        # 50% de 1.0 et 50% de réels arbitraires alternant aléatoirement
        # -> pire cas pour le prédicteur de branchement du CPU !
        for i in 1:n
            f.u_pivot_value[i] = rand(rng, Bool) ? 1.0 : (0.1 + 2.0 * rand(rng))
        end
    end

    # Calcul vectorisé des inverses
    resize!(f.u_pivot_inv_value, n)
    @inbounds @simd for i in 1:n
        f.u_pivot_inv_value[i] = 1.0 / f.u_pivot_value[i]
    end

    return f
end

function benchmark_strategy_solves(f::HFactor, strategy::PivotStrategy, repeats::Int)
    n = f.num_row
    rhs = HVector(n)
    val_strategy = Val(strategy)

    # Warmup
    for _ in 1:10
        rhs.count = min(20, n)
        for k in 1:rhs.count
            rhs.index[k] = k
            rhs.array[k] = 1.25 * k
        end
        TinyHiGHS.solveHyper!(rhs, n, f.u_pivot_lookup, f.u_pivot_index,
            f.u_pivot_value, true, f.u_start, f.u_last_p, 0, f.u_index,
            f.u_value, f.u_pivot_inv_value, val_strategy)
    end

    t0 = time_ns()
    for _ in 1:repeats
        rhs.count = min(20, n)
        for k in 1:rhs.count
            rhs.index[k] = k
            rhs.array[k] = 1.25 * k
        end
        TinyHiGHS.solveHyper!(rhs, n, f.u_pivot_lookup, f.u_pivot_index,
            f.u_pivot_value, true, f.u_start, f.u_last_p, 0, f.u_index,
            f.u_value, f.u_pivot_inv_value, val_strategy)
    end
    t1 = time_ns()

    return (t1 - t0) / repeats  # nanosecondes par appel
end

function run_all_profiles(n::Int=1000, repeats::Int=50_000)
    println("================================================================================")
    println(" Micro-benchmark Micro-Architectural des Stratégies de Pivot (solveHyper)")
    @printf(" Dimension n = %d, Répétitions = %d appels par mesure\n", n, repeats)
    println("================================================================================")

    profiles = [
        (:network, "1. Profil Réseau (90% unitaires ±1.0, prédiction ~95%)"),
        (:mixed_mispredict, "2. Profil Mixte Imprévisible (50% unitaires / 50% réels -> échec TAGE)"),
        (:general, "3. Profil Général / Continu (100% réels arbitraires, Netlib/Mittelmann)")
    ]

    for (mode, desc) in profiles
        f = make_synthetic_hfactor(n, mode)
        println("\n--- $desc ---")

        # Mesures A/B entrelacées
        t_branchless = Float64[]
        t_branching = Float64[]
        t_fdiv = Float64[]

        for _ in 1:7
            push!(t_branchless, benchmark_strategy_solves(f, kPivotBranchless, repeats))
            push!(t_branching, benchmark_strategy_solves(f, kPivotBranching, repeats))
            push!(t_fdiv, benchmark_strategy_solves(f, kPivotFdiv, repeats))
        end

        min_bl = minimum(t_branchless)
        min_br = minimum(t_branching)
        min_fd = minimum(t_fdiv)

        @printf("  kPivotBranchless (* inv_pivot) : %6.1f ns / solve   [base 1.00x]\n", min_bl)
        @printf("  kPivotBranching  (if ±1.0)     : %6.1f ns / solve   [%+.1f%% vs branchless]\n",
            min_br, ((min_br - min_bl) / min_bl) * 100)
        @printf("  kPivotFdiv       (/ pivot)     : %6.1f ns / solve   [%+.1f%% vs branchless]\n",
            min_fd, ((min_fd - min_bl) / min_bl) * 100)
    end
    println("\n================================================================================")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_profiles()
end
