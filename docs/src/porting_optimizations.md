# From C++ to Zero-Allocation Julia: Evolution & Optimizations

This document details the engineering journey of **TinyHiGHS.jl**: how it evolved from a literal line-by-line transcription of HiGHS's C++ source code into a highly optimized, idiomatic, zero-allocation pure Julia solver.

---

## 1. Motivation: The High-Frequency LP Bottleneck

In numerical optimization paradigms such as **Stochastic Dual Dynamic Programming (SDDP)**, **Benders Decomposition**, and **Column Generation**, algorithms solve hundreds of thousands (or millions) of small, closely related linear subproblems.

In this regime:
1. **FFI Overhead**: Calling a C/C++ solver via Foreign Function Interface (`ccall`) incurs memory copying and boundary crossing overhead.
2. **Dynamic Memory Allocation**: In standard C++ solvers, each solve (even when warm-started) reallocates and resizes internal `std::vector` buffers across solver layers.
3. **Latency Matters More Than Throughput**: The bottleneck shifts from matrix factorization complexity to memory allocation latency, cache locality, and function call overhead.

**Goal of TinyHiGHS.jl**: Achieve microsecond-level solve times (under 50 µs) with **strictly zero heap allocations** during repeated resolves, while retaining a **bit-for-bit reference path** compatible with HiGHS. The optional branchless reciprocal path is validated with a one-ULP bound for arbitrary pivots.

---

## 2. Phase 1: The Literal C++ Port & Numerical Oracle

The initial development stage focused on algorithmic correctness and numerical fidelity.

### Strict Bit-for-Bit Validation Against Frozen Oracle
Every mathematical kernel was mirrored and compared against a frozen, instrumented C++ reference oracle compiled from HiGHS 1.15.1 (`04024d701f79feb8e2f18bc3df0dffc04ef05088`):
- `HVector` sparse operations (`tight!`, `pack!`, `saxpy!`, `norm2`).
- `HFactor` LU factorization with Markowitz search and Forrest-Tomlin updates.
- `SparseMatrix` column-wise, row-wise, and partitioned pricing.
- `HEkkDual` and `HEkkPrimal` ratio tests, pricing, and iteration loops.

```mermaid
flowchart LR
    A[HiGHS C++ Oracle] -->|Same random seeds & instances| C[Bit-for-bit Exact IEEE-754 Equality]
    B[TinyHiGHS.jl] -->|Same operations & tie-breaking| C
    C -->|0 ULP gap| D[Validated Correctness]
```

### Challenges of the Literal Port
A direct C++ to Julia translation presents specific pitfalls:
- **0-based vs 1-based indexing**: C++ uses 0-based indices with negative sentinels (`-1` for unset). A naive shift by `+1` everywhere introduces subtle branch bugs. TinyHiGHS systematically shifted structural positions to 1-based while maintaining encoded values and list counters.
- **Shared Pointer Semantics**: In C++, `HSimplexNla` and `HFactor` store raw pointers to `basic_index` owned by `HEkk`. In Julia, this was achieved by having all structures share the exact same `Vector{Int}` reference, ensuring permutations applied in `HFactor.build!` immediately reflect across the entire engine.

---

## 3. Phase 2: Key Architectural Innovations in TinyHiGHS

Once correctness was established, the codebase was systematically re-architected to exploit Julia's high-performance capabilities and eliminate micro-architectural bottlenecks.

### A. Zero-Allocation Persistent Buffer Architecture

#### The Problem in Upstream C++
Profiling upstream HiGHS during warm-start sequences revealed significant hidden allocations:
- Calling `highs.run()` or modifying bounds invokes routines that instantiate temporary vectors and resize buffers in `HEkk` and `HFactor`.
- While negligible for large single solves, in high-frequency warm-start sequences (e.g., 100 solves in 15 ms), memory allocators dominate execution time.

#### The TinyHiGHS Solution
In TinyHiGHS:
1. All working arrays in `SimplexInfo` and `HVector` workspaces (`row_ep`, `row_ap`, `col_aq`, `col_BFRT`) are allocated **once** during `SimplexEngine` construction, sized to the problem capacity (`num_col + num_row`).
2. Subsequent in-place modifications (`change_col_bounds!`, `change_row_bounds!`, `change_cols_cost!`) update vectors directly and mark dirty flags without allocating new memory.
3. Resolves execute with **0 bytes allocated** and **0 GC cycles**.

```mermaid
graph TD
    subgraph "Cold Initialization (Once)"
        Alloc[Preallocate buffers up to num_col + num_row]
        Build[Initial LU Factorization]
    end
    subgraph "Warm-Start Sequence (Repeated N times)"
        Mod[change_col_bounds! in-place: 0 bytes]
        Solve[solve!: in-place LU update + resolve: 0 bytes]
        Sol[Solution extraction: 0 bytes]
        Mod --> Solve --> Sol --> Mod
    end
    Alloc --> Build --> Mod
```

---

### B. Micro-Architectural Optimization: SIMD Reciprocal and Branchless Substitution

The current HiGHS experiment does not branch on unit pivots. Instead, every
diagonal pivot is handled by the same reciprocal path:

1. `HFactor` computes `1.0 / pivot` once during factorization in a contiguous,
   compiler-vectorized pass.
2. Forrest–Tomlin updates append the reciprocal for each new diagonal pivot.
3. `ftranU`, `btranU`, and `solveHyper` multiply by the stored reciprocal,
   replacing the per-pivot `FDIV` in the triangular solves.

This keeps the hot loop branchless and makes the optimization useful for mixed
matrices, not only network matrices dominated by `±1` coefficients. The
reciprocal path is bit-for-bit exact for `±1` and powers of two; for arbitrary
non-zero pivots the tested difference is bounded by one ULP. TinyHiGHS therefore
retains `kPivotBranching` as the strict frozen-oracle path and exposes
`kPivotBranchless` for the reciprocal experiment.

The implementation is available in the HiGHS branch
`perf/simd-branchless-pivots`, with the warm-start regression test and replay
instructions documented in the repository's
[`contrib_highs/README.md`](https://github.com/LaurentPlagne/TinyHIGHS.jl/blob/main/contrib_highs/README.md).

---

### C. Idiomatic Julia & Type Stability

Unlike C++ which relies on templates, inheritance, and macros, TinyHiGHS embraces modern Julia paradigms:
1. **Full Type Stability (`@inferred`)**: Every core solver function has concrete, predictable return types. Zero runtime boxing, zero dynamic dispatch in the inner loop.
2. **Inlined Dense Dot Products**: Vectorized dot products (`compute_dot`, `alpha_product_plus_y!`) are inlined and decorated with `@inbounds @simd` for automatic SIMD vectorization.
3. **No Splatting in Critical Paths**: Replaced tuple splatting and temporary generator allocations with direct array accesses.

---

### D. Standalone Native LP I/O

The original C++ port was coupled to internal data loaders. TinyHiGHS features an entirely autonomous CPLEX `.lp` parser and serializer implemented in **100% pure Julia stdlib**:
- The core has no mandatory dependency on MathOptInterface, JuMP, or external
  parsers; an optional MOI extension is provided for standard modeling tools.
- Reads `.lp` files directly into CSC matrix structures.
- Allows immediate standalone benchmarking and testing.

---

## 4. Summary of Changes: Literal Port vs. TinyHiGHS.jl

| Feature / Aspect | Initial Literal Port | TinyHiGHS.jl |
| :--- | :--- | :--- |
| **Language & Dependencies** | C++ dependent (requires compiler & oracle) | **100% Pure Julia, zero dependencies** |
| **Warm-Start Resolves** | Reallocated temporary arrays | **Strictly 0 bytes allocated** |
| **FTRAN / BTRAN Pivots** | Unconditional `val /= pivot` (per-pivot `FDIV`) | **Precomputed reciprocal `val *= pivot_inverse` (branchless)** |
| **LP File I/O** | External / C++ test harnesses | **Native `read_lp` / `write_lp` in Julia stdlib** |
| **Type Inférence** | Partial `Union` returns | **100% `@inferred` type stable** |
| **Solve Time (`sequence_small`)** | ~223 µs / solve | **32.4 µs / solve (6.9x faster)** |
| **Numerical Consistency** | Exact IEEE-754 match | **Exact for $\pm1$ / powers of two; branchless path <=1 ULP on arbitrary pivots** |
