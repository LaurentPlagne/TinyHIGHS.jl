# Upstream Contribution Proposal for HiGHS C++

This document summarizes performance optimizations identified, validated, and benchmarked during the development of **[TinyHiGHS.jl](https://github.com/LaurentPlagne/TinyHIGHS.jl)** (a faithful, zero-allocation pure Julia port of HiGHS's dual and primal revised simplex solvers).

All optimizations preserve **strict bit-for-bit IEEE-754 equivalence** against frozen HiGHS reference oracles.

---

## 1. Short-Circuit Unit Diagonal Pivots in FTRAN / BTRAN (`HFactor`)

### Motivation & Empirical Finding
In network flow, transportation, scheduling, and multi-commodity problems, constraint matrices are predominantly composed of $\{0, \pm 1\}$ coefficients. During LU factorization with Forrest-Tomlin updates:
- On real-world network instances, **82.8% of the diagonal pivots in $U$ are exactly $+1.0$**, and **7.3% are $-1.0$**.
- Over **90.1% of all floating-point divisions during FTRAN and BTRAN are divisions by $\pm 1.0$**!

### Micro-architectural Impact
On modern x86-64 and AArch64 (Apple Silicon / Neoverse / AMD Zen / Intel Core) architectures:
- Floating-point division (`FDIV` / `vdivsd`) carries a latency of **10 to 15 clock cycles** and cannot be pipelined at the same rate as addition or multiplication.
- Checking `pivot == 1.0` and `pivot == -1.0` allows the branch predictor (with a 90% hit rate) to replace an expensive 15-cycle division with a 0-cycle pass-through or a 1-cycle negation.
- Because $x / 1.0 \equiv x$ and $x / (-1.0) \equiv -x$ in IEEE-754 arithmetic (for all finite and non-NaN floats), this optimization is **numerically exact and bit-for-bit identical to division**.

### Concrete Benchmarks: Native HiGHS C++ (Before vs. After Patch)
The patch was applied directly to **HiGHS 1.15.1** (`highs/util/HFactor.cpp`) and compiled with `-O3 -DNDEBUG` on Apple Silicon:

#### A. Single Cold-Start Solves (`instances/benchmarks/`)
| Instance | HiGHS C++ Original | HiGHS C++ Patched | Speedup / Gain | Simplex Iterations | Objective Gap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `netflow_medium_02.lp` | 9.69 ms | **7.62 ms** | **+21.4 %** | 390 vs 390 | **0.0 (exact)** |
| `netflow_large_01.lp` | 9.71 ms | **7.62 ms** | **+21.5 %** | 291 vs 291 | **0.0 (exact)** |
| `netflow_small_02.lp` | 5.69 ms | **3.75 ms** | **+34.0 %** | 46 vs 46 | **0.0 (exact)** |
| `netflow_small_01.lp` | 5.69 ms | **3.65 ms** | **+35.9 %** | 49 vs 49 | **0.0 (exact)** |

#### B. Warm-Start Sequences (100% Native C++ API Replay, Zéro Julia)
Replayed through `contrib_highs/cpp/replay_sequence.cpp` using official `Highs` C++ API:

| Sequence (`instances/sequences/`) | Solves | HiGHS C++ Original | HiGHS C++ Patched | Speedup | Iterations | Final Obj Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sequence_small` | 76 | 16.97 ms (223 µs/solve) | **4.10 ms (54 µs/solve)** | **4.14x (+75.8%)** | 47 vs 47 | **0.0 (exact)** |
| `sequence_medium` | 100 | 182.50 ms (1.82 ms/solve) | **148.27 ms (1.48 ms/solve)** | **1.23x (+18.8%)** | 549 vs 549 | **0.0 (exact)** |

---

## 2. Rejouer le Benchmark C++ en Local (Zéro Julia)

Un script autonome bash + C++11 est fourni pour rejouer l'intégralité du banc d'essai sans aucune dépendance Julia :

```bash
cd TinyHiGHS.jl
./contrib_highs/run_bench_cpp.sh
```

Ce script :
1. Compile l'exécutable C++ `contrib_highs/cpp/replay_sequence.cpp` avec `clang++ -O3 -std=c++11`.
2. Exécute le comparatif A/B (HiGHS officiel v1.15.1 vs HiGHS patché).
3. Affiche les temps moyens par solve (en microsecondes), les itérations du simplexe et la stricte égalité d'objectif.

---

## 3. Contenu de la Pull Request pour HiGHS

La PR est prête sur la branche `perf/unit-diagonal-pivots` du dépôt local de HiGHS. Elle contient :
1. **L'optimisation du noyau** :
   - `highs/util/HFactor.cpp` : court-circuit des divisions $\pm 1$ dans `ftranU`, `btranU` et `solveHyper`.
2. **Le cas de test unitaire Catch2 officiel** :
   - `check/TestSequenceWarmStart.cpp` : test complet de warm-start avec 240 assertions validées.
   - `check/CMakeLists.txt` : enregistrement dans la cible `unit_tests`.
3. **Les assets de test (anonymisés)** :
   - `check/instances/sequence_small_base.lp` (12 Ko)
   - `check/instances/sequence_small_operations.txt` (4 Ko)

### Procédure pour soumettre la PR sur GitHub :
1. Forker `https://github.com/ERGO-Code/HiGHS` sur votre compte GitHub.
2. Dans le clone local de HiGHS (`third_party/solvers/src/HiGHS`) :
   ```bash
   git remote add myfork git@github.com:<votre-compte>/HiGHS.git
   git push -u myfork perf/unit-diagonal-pivots
   ```
3. Ouvrir la PR sur GitHub vers `ERGO-Code/HiGHS` (branche `latest`).

---

## 4. Perspectives : Réduction des allocations et architecture de tampons persistants

Lors des benchmarks sur les séquences d'optimisation stochastique, **TinyHiGHS.jl atteint 32.4 µs par solve** (soit encore 1.6x plus rapide que HiGHS C++ patché à 54 µs) :
- **Cause de l'écart** : Dans HiGHS C++, l'appel à `highs.run()` et aux fonctions de modification de bornes réalloue ou redimensionne dynamiquement de multiples conteneurs `std::vector` à travers la hiérarchie `Highs` -> `HEkk` -> `HFactor`.
- **Piste proposée pour HiGHS 2.x** : Introduire un mode d'exécution à « tampons persistants » où l'espace de travail est alloué une seule fois à capacité maximale et réutilisé sans aucune allocation tas entre résolutions successives en warm-start.
- **Référence publique** : L'implémentation complète et documentée de ce mécanisme à zéro allocation est consultable publiquement dans le dépôt open-source [TinyHiGHS.jl](https://github.com/LaurentPlagne/TinyHIGHS.jl).
