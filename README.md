# TinyHiGHS.jl

[![CI](https://github.com/laurentplagne/TinyHiGHS.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/laurentplagne/TinyHiGHS.jl/actions/workflows/CI.yml)
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
git clone https://github.com/laurentplagne/TinyHiGHS.jl.git
cd TinyHiGHS.jl
```

### 1. Warm-Start Sequence Benchmark (TinyHiGHS vs HiGHS C++)
Replays continuous sequences of bound modifications and warm-start resolves on a persistent engine:

```bash
julia --project=. bench/compare_sequences.jl
```

**Measured Results (Apple Silicon / AArch64):**
| LP Sequence (`instances/sequences/`) | Solves | Dimensions | TinyHiGHS.jl (Total) | HiGHS C++ (Total) | Speedup | TinyHiGHS Time/Solve |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sequence_small`** | 76 | $88 \times 107$ | **2.47 ms** | 28.71 ms | **11.6x** | **32.4 µs / solve** |
| **`sequence_medium`** | 100 | $1\,240 \times 1\,483$ | **61.79 ms** | 140.21 ms | **2.3x** | **617.9 µs / solve** |

### 2. Cold-Start Individual LPs (`.lp` files)
Solves individual benchmark instances and compares bit-for-bit against the official HiGHS C++ CLI:

```bash
julia --project=. bench/compare_highs.jl
```

### 3. Pure C++ Standalone Replay (Zero Julia Required)
You can also replay the sequences directly in 100% native C++ using official HiGHS headers and libraries:

```bash
./contrib_highs/run_bench_cpp.sh
```

---

## 🔬 Key Architectural Findings & Upstream HiGHS Contributions

The development and profiling of TinyHiGHS revealed two major micro-architectural insights:

### 1. Short-Circuiting Unit Diagonal Pivots in `HFactor`
In network-flow, multi-commodity, and scheduling problems, incidence matrices are dominated by $\{0, \pm 1\}$ coefficients.
* **Empirical observation**: In real network sequences, **over 90% of all diagonal pivots in $U$ are exactly $\pm 1.0$**.
* **Micro-architectural gain**: Floating-point division (`vdivsd` / `FDIV`) has a latency of 10–15 CPU cycles. Checking for `pivot == 1.0` and `pivot == -1.0` bypasses division with a 0-cycle pass-through or a 1-cycle negation. Because IEEE-754 division by $\pm 1.0$ is exact for finite numbers, this optimization is **numerically bit-for-bit identical**.
* **Upstream impact on HiGHS C++**: Applied directly to `HFactor.cpp` in HiGHS 1.15.1, this yields a **+21% to +36% speedup** on cold-start CLI solves and up to **4.14x speedup** on warm-start sequences in native C++!
* **Upstream PR**: A clean Pull Request with Catch2 unit tests has been prepared for `ERGO-Code/HiGHS`. See [`contrib_highs/README.md`](contrib_highs/README.md) for technical details and instructions.

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
