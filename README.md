# TinyHiGHS.jl

[![CI](https://github.com/LaurentPlagne/TinyHIGHS.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LaurentPlagne/TinyHIGHS.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/LaurentPlagne/TinyHIGHS.jl/actions/workflows/Documentation.yml/badge.svg)](https://LaurentPlagne.github.io/TinyHIGHS.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TinyHiGHS.jl** is a lightweight, zero-allocation, pure-Julia experimental port of the **dual and primal revised simplex engines** from [HiGHS](https://highs.dev/) (the premier open-source LP/MIP solver developed at the University of Edinburgh).

It is designed as an **algorithmic and micro-architectural research laboratory**: a sandbox to prototype, benchmark, and measure low-level optimizations (zero-allocation persistent buffers, sparse LU update paths, and division elimination) with the primary goal of **contributing performance improvements back upstream to HiGHS C++**.

---

## 🎯 Scope & Purpose

TinyHiGHS is **not** a general-purpose replacement for HiGHS. Its scope is deliberately focused on high-frequency, sequential linear programming where cold-start overhead and memory churn dominate.

### What is in scope:
- **Dual Revised Simplex (`HEkkDual`)**: Dantzig pricing, Steepest-Edge / Devex weights, dual perturbation, and bound flips (BFRT).
- **Primal Revised Simplex (`HEkkPrimal`)**: Direct primal phase 1 / phase 2 execution.
- **Sparse LU Factorization (`HFactor`)**: Markowitz LU factorization with Forrest-Tomlin updates, hyper-sparse and sparse FTRAN / BTRAN passes.
- **Persistent Engine & Zero-Allocation Warm-Start**: Persistent workspace buffers where solving sequences of modified LPs executes with **0 bytes allocated on the heap**.
- **Self-Contained & Zero Dependencies**: 100% pure Julia stdlib (`Test`, `Random`, `Printf`), built-in lightweight `.lp` reader and writer.

### What is out of scope (use upstream [HiGHS](https://github.com/ERGO-Code/HiGHS) instead):
- Mixed-Integer Linear Programming (MIP branch-and-cut).
- Interior Point Methods (IPX, barrier, crossover).
- First-order solvers (PDLP).
- Quadratic programming (QP).
- Heavy presolve reductions and parallel simplex.

---

## ⚡ Quickstart: Run Benchmarks in 1 Minute

Everything is self-contained. Clone and run:

```bash
git clone https://github.com/LaurentPlagne/TinyHIGHS.jl.git
cd TinyHIGHS.jl
```

### 1. Warm-Start Sequence Benchmark (TinyHiGHS vs HiGHS C++)
Replays continuous sequences of bound modifications and warm-start resolves on a persistent engine:

```bash
julia --project=. bench/compare_sequences.jl
```

To regenerate the checked-in sequence snapshots from their operation logs:

```bash
julia --project=. bench/regenerate_sequence_assets.jl
```

**Measured Results (Apple Silicon / AArch64):**
| LP Sequence (`instances/sequences/`) | Solves | Dimensions | TinyHiGHS.jl (Total) | HiGHS C++ (Total) | Speedup | TinyHiGHS Time/Solve |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sequence_small`** | 76 | $88 \times 107$ | **2.47 ms** | 28.71 ms | **11.6x** | **32.4 µs / solve** |
| **`sequence_medium`** | 100 | $1\,240 \times 1\,483$ | **61.79 ms** | 140.21 ms | **2.3x** | **617.9 µs / solve** |

### 2. Cold-Start Individual LPs (`.lp` files)
Solves individual benchmark instances and compares the default reference strategy
bit-for-bit against the official HiGHS C++ CLI:

```bash
julia --project=. bench/compare_highs.jl
```

### 3. Pure C++ Standalone Replay (Zero Julia Required)
You can also replay the sequences directly in 100% native C++ using official HiGHS headers and libraries:

```bash
./contrib_highs/run_bench_cpp.sh
```

### 4. Full Three-Way Benchmark

The exact command block used for the HiGHS upstream discussion is also kept in
[`contrib_highs/OSCAR_MESSAGE.md`](contrib_highs/OSCAR_MESSAGE.md):

```bash
git clone https://github.com/LaurentPlagne/TinyHIGHS.jl.git
cd TinyHIGHS.jl
./contrib_highs/run_bench_cpp.sh
julia --project=. bench/bench_3way.jl
```

The Julia benchmark reports unavailable C++ runners as `N/A` rather than
failing, and marks objective mismatches as `DIFF` so performance numbers are
never mistaken for a correctness validation.

### 5. Configuring HiGHS C++ Location (Custom Build vs Julia Artifact)

By default, the native runner first looks for a sibling `HiGHS/build` checkout
(the recommended way to test the patch), then checks the paths below, and finally
falls back to the official HiGHS artifact shipped via Julia. It recompiles the
small C++ driver on every invocation and prints the selected library paths, so a
stale binary cannot silently turn the A/B comparison into two runs of the same
library.

However, if you wish to benchmark against a **custom C++ build** (such as your local clone with the `HFactor` optimization patch applied):

```bash
# Point to a HiGHS source checkout containing build/lib/libhighs:
export HIGHS_DIR=/path/to/HiGHS

# Or point directly to its CMake build directory:
export HIGHS_BUILD_DIR=/path/to/HiGHS/build

# Or point to a custom installation prefix:
export HIGHS_INSTALL=/path/to/HiGHS/install

# For the Julia-to-CLI comparison only, specify the highs executable:
export HIGHS_BIN=/path/to/HiGHS/build/bin/highs

# Then run the benchmark as usual:
julia --project=. bench/compare_highs.jl
```

---

## 🔬 Key Architectural Findings & Upstream HiGHS Contributions

The development and profiling of TinyHiGHS revealed two major micro-architectural insights:

### 1. SIMD Reciprocal and Branchless Diagonal Substitution in `HFactor`
The current upstream experiment replaces the per-pivot floating-point division in
`ftranU`, `btranU`, and `solveHyper` with a multiplication by a precomputed
reciprocal.
* **Pre-inversion**: diagonal pivots are inverted once during factorization (and
  when Forrest–Tomlin updates append a pivot) in a contiguous, compiler-vectorized
  pass.
* **Branchless substitution**: the solve kernels use
  `pivot_multiplier *= u_pivot_inv_value[i]` for every pivot, avoiding the
  unpredictable `FDIV`/branch trade-off on mixed matrices.
* **Numerical contract**: there is no unit-pivot special case in the hot loop.
  Multiplication by the reciprocal is bit-for-bit exact for `±1` and powers of
  two; arbitrary pivots are bounded by one ULP in the tested corpus. TinyHiGHS
  keeps `kPivotBranching` as the strict reference path and exposes
  `kPivotBranchless` for the reciprocal experiment.
* **Upstream PR**: the corresponding HiGHS branch is
  `perf/simd-branchless-pivots`. See [`contrib_highs/README.md`](contrib_highs/README.md)
  for implementation and reproduction details.

### 2. Persistent Buffer Architecture (Zero Heap Allocations)
In standard HiGHS C++, calling `highs.run()` or modifying problem bounds triggers multiple buffer resizes, memory reallocations, and state copies across the `Highs` -> `HEkk` -> `HFactor` hierarchy.
* By contrast, TinyHiGHS keeps a single persistent workspace where scratch arrays and factorizations grow up to capacity and are cleared in-place.
* This eliminates memory allocator churn, yielding solve times as low as **32 µs per warm-start resolve**.
* We propose this persistent buffer pattern as a roadmap item for future high-performance warm-start modes in upstream HiGHS.

---

## 🔒 Confidentiality & Anonymization

All instances in `instances/` (both standalone `.lp` models and warm-start operation logs) are **fully anonymized**:
* Columns are systematically renamed `c0`, `c1`, `c2`, ...
* Rows are systematically renamed `r0`, `r1`, `r2`, ...
* All domain-specific, geographical, corporate, or project names have been completely scrubbed.
* Git commit history is clean and free of proprietary references.

---

## 👥 Authors & Acknowledgments

- **Port & Architecture**: Laurent Plagne & contributors.
- **Original HiGHS Solvers & Theory**: Julian Hall, Ivet Galabova, Leona Gottwald, and the HiGHS team ([ERGO-Code](https://github.com/ERGO-Code/HiGHS), School of Mathematics, University of Edinburgh).

## 📄 License

TinyHiGHS.jl is released under the [MIT License](LICENSE).
