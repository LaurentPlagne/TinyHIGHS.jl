# TinyHiGHS.jl

*High-performance, zero-allocation revised simplex solver in 100% pure Julia.*

[![CI](https://github.com/LaurentPlagne/TinyHIGHS.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LaurentPlagne/TinyHIGHS.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/LaurentPlagne/TinyHIGHS.jl/actions/workflows/Documentation.yml/badge.svg)](https://LaurentPlagne.github.io/TinyHIGHS.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**TinyHiGHS.jl** is a faithful, standalone, zero-allocation pure Julia implementation of the dual and primal revised simplex solvers from the state-of-the-art **[HiGHS](https://github.com/ERGO-Code/HiGHS)** linear programming library (MIT License).

It is specifically architected for scenarios where linear programs must be solved repeatedly in tight computational loops—such as **Stochastic Dual Dynamic Programming (SDDP)**, **Benders decomposition**, **Branch-and-Price**, and **Network Flow simulations**—where standard FFI overhead and dynamic memory allocation become severe performance bottlenecks.

```mermaid
graph TD
    A[Input: .lp file or Julia SimplexLp] --> B[SimplexEngine]
    B --> C[SparseMatrix & Scaler]
    B --> D[HFactor: LU Factorization + Forrest-Tomlin]
    B --> E[HVector: Hyper-Sparse Workspaces]
    B --> F{Simplex Algorithm}
    F -->|Default| G[DualSolver: DSE / Devex / BFRT]
    F -->|Alternative| H[PrimalSolver: Phase 1 & 2 / Harris Ratio]
    G --> I[Optimal Solution: 0 Allocations in Resolves]
    H --> I
```

---

## Key Highlights

- 🚀 **Zero Allocations in Warm-Start Resolves**: Once initialized, repeated resolves with modified bounds or objective costs require **0 bytes allocated** and execute in **30 to 50 microseconds** per solve.
- ⚡ **Branchless Reciprocal Substitution**: Pre-inverts the diagonal pivots in
  `HFactor` with a compiler-vectorized pass and replaces per-pivot `FDIV` with a
  multiplication in FTRAN/BTRAN and `solveHyper`.
- 💎 **Reference Numerical Accuracy**: The default `kPivotBranching` strategy is checked for strict IEEE-754 equivalence against frozen HiGHS reference oracles. The optional branchless reciprocal strategy is tested with an explicit one-ULP bound for arbitrary pivots.
- 📦 **100% Pure Julia & Zero Dependencies**: Runs out-of-the-box on macOS (Apple Silicon & Intel), Linux (x86-64 & AArch64), and Windows without requiring any external C/C++ shared library, CMake, or compiler toolchain.
- 📝 **Native CPLEX `.lp` Reader & Writer**: Fast, dependency-free text I/O compatible with HiGHS, CPLEX, Gurobi, and Clp.

---

## Quick Example

```julia
using TinyHiGHS

# 1. Load an LP instance from a .lp file
lp = read_lp("instances/benchmarks/netflow_small_01.lp")

# 2. Solve using the high-level API
status, obj, engine = solve_lp(lp)

println("Status: ", status)
println("Objective: ", obj)

# 3. High-performance warm-start: modify bounds and resolve with 0 allocations!
change_col_bounds!(engine, 1, 0.0, 50.0)
status_warm = solve!(engine) # 0 bytes allocated after warm-up; measure locally
```

---

## Performance Snapshot

| Benchmark Instance | Solves | HiGHS_artifact | HiGHS_branchless | TinyHiGHS branchless | Speedup vs artifact | Allocations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sequence_small` (warm-start) | 76 | 21.51 ms | 5.41 ms | **3.80 ms (50.0 µs/solve)** | **5.66x** | **0 bytes** |
| `sequence_medium` (warm-start) | 100 | 269.25 ms | 222.33 ms | **160.71 ms (1.61 ms/solve)** | **1.68x** | **0 bytes** |

This is a reproducible snapshot from `julia --project=. bench/bench_3way.jl`
(warm-start) on Apple Silicon. The cold-start table is maintained on the
[Benchmark Suite](benchmarks.md) page. Timings vary with hardware and system
load; rerun the commands for fresh values.

See the [Benchmarks](benchmarks.md) and [From C++ to Zero-Allocation Julia](porting_optimizations.md) sections for complete details and reproduction scripts.

---

## Manual Table of Contents

- [Quick Start Guide](quickstart.md): Installation, model formulation, solving, and inspecting solutions.
- [System Architecture](architecture.md): Deep-dive into `SimplexEngine`, `HFactor`, `HVector`, and solver algorithms.
- [From C++ to Zero-Allocation Julia](porting_optimizations.md): Evolution, architectural enhancements, branchless substitution, and memory layout.
- [Benchmark Suite](benchmarks.md): Methodology, reproducibility, and comparative analysis.
- [API Reference](api.md): Complete reference for all exported types, functions, and control options.
