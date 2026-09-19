# Benchmark Suite

This page documents the performance benchmarks, test instances, and instructions for full reproducibility.

---

## Benchmark Scripts

TinyHiGHS includes four standalone benchmark harnesses plus a corpus-generation
utility:

### 1. Cold-Start Benchmarks (Julia vs. HiGHS CLI)
Compares single cold-start resolutions of `.lp` instances between TinyHiGHS and the official HiGHS binary:

```bash
julia --project=. bench/compare_highs.jl
```

### 2. Warm-Start Sequence Benchmarks (SimplexEngine In-Place)
Simulates realistic stochastic optimization sequences (successive resolves with updated variable bounds) and profiles execution time and memory allocations:

```bash
julia --project=. bench/compare_sequences.jl
```

### 3. Standalone Native C++ Replay (Zero Julia)
Compares the C++ implementation of official HiGHS against the patched HiGHS C++ kernel (with the unit pivot short-circuit) using a pure C++ runner:

```bash
./contrib_highs/run_bench_cpp.sh
```

### 4. Netlib correctness and timing smoke suite

The curated Netlib corpus is solved from Julia and checked against published
optimal objectives. The optional C++ comparison is skipped when no `highs`
executable is installed:

```bash
julia --project=. bench/netlib_benchmarks.jl
```

### 5. Reproducible warm-start corpus generation

The checked-in sequence snapshots are generated from `base.lp` and
`operations.txt`.  The generator rewrites the base model with explicit
`c0`, `c1`, ... objective terms, then replays every operation and verifies an
optimal status before writing the first ten snapshots:

```bash
julia --project=. bench/regenerate_sequence_assets.jl
```

This explicit column order is required because replay logs address columns by
numeric position, while a generic LP reader may otherwise discover
objective-only variables before variables appearing in constraints.

---

## Detailed Results

All benchmarks were recorded on Apple Silicon (M-series, AArch64) using Clang with `-O3 -DNDEBUG` and Julia v1.12+.

### A. Warm-Start Sequences

A sequence of consecutive solves where bounds on variables are dynamically updated between resolves, simulating decomposition subproblem evaluations:

| Sequence | Solves | HiGHS C++ (1.15.1) | HiGHS C++ Patched | TinyHiGHS.jl | TinyHiGHS Speedup | Allocations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sequence_small` | 76 | 16.97 ms (223 µs/solve) | 4.10 ms (54 µs/solve) | **2.46 ms (32.4 µs/solve)** | **6.9x faster** | **0 bytes** |
| `sequence_medium` | 100 | 182.50 ms (1.82 ms/solve) | 148.27 ms (1.48 ms/solve) | **141.30 ms (1.41 ms/solve)** | **1.3x faster** | **0 bytes** |

### B. Single Cold-Start Solves

Solving network flow instances from scratch (including initial basis setup and full factorization):

| Instance | Matrix Dimensions | HiGHS C++ CLI | TinyHiGHS.jl | Speedup | Iterations | Obj Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `netflow_small_01.lp` | $56 \times 70$ | 5.69 ms | **3.51 ms** | **+38.3 %** | 49 vs 49 | **0.0 (exact)** |
| `netflow_small_02.lp` | $56 \times 70$ | 5.69 ms | **3.56 ms** | **+37.4 %** | 46 vs 46 | **0.0 (exact)** |
| `netflow_medium_02.lp` | $256 \times 320$ | 9.69 ms | **7.42 ms** | **+23.4 %** | 390 vs 390 | **0.0 (exact)** |
| `netflow_large_01.lp` | $512 \times 640$ | 9.71 ms | **7.54 ms** | **+22.3 %** | 291 vs 291 | **0.0 (exact)** |

---

## Reproducibility Checklist

- **Correctness is explicit**: the native C++ replay checks equal objectives and
  iteration counts. The three-way Julia benchmark labels each TinyHiGHS result
  `OK`, `DIFF`, or `NONOPT`; a speedup is not presented as a correctness claim.
- **Zero JIT Bias**: Warm-start benchmarks run multiple dry iterations to precompile Julia methods before measuring steady-state resolve latency.
- **Hardware Isolation**: Measurements were repeated with CPU governor pinned to avoid frequency scaling artifacts.
