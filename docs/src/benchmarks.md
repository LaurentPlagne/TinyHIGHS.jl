# Benchmark Suite

This page documents the performance benchmarks, test instances, and instructions for full reproducibility.

---

## Benchmark Scripts

TinyHiGHS includes five standalone benchmark harnesses plus a corpus-generation
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
Compares the official HiGHS artifact against the local `HiGHS_branchless` C++
kernel (SIMD reciprocal pre-inversion and branchless substitution) using a pure
C++ runner:

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

The snapshot below was recorded on Apple Silicon (M-series, AArch64) with Clang
`-O3 -DNDEBUG` and Julia v1.12+. Absolute timings vary with hardware and system
load; the commands above are the reproducibility contract.

### A. Warm-Start Sequences

A sequence of consecutive solves where bounds on variables are dynamically updated between resolves, simulating decomposition subproblem evaluations:

| Sequence | Solves | HiGHS artifact (1.15.1) | HiGHS_branchless | TinyHiGHS branching | TinyHiGHS branchless | TinyHiGHS FDIV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sequence_small` | 76 | 21.51 ms (283.0 µs/solve) | 5.41 ms (71.2 µs/solve, 3.98x) | **4.06 ms (53.5 µs/solve, 5.29x)** | **3.80 ms (50.0 µs/solve, 5.66x)** | 3.92 ms (51.6 µs/solve, 5.49x) |
| `sequence_medium` | 100 | 269.25 ms (2692.5 µs/solve) | 222.33 ms (2223.3 µs/solve, 1.21x) | **159.71 ms (1597.1 µs/solve, 1.69x)** | **160.71 ms (1607.1 µs/solve, 1.68x)** | 160.29 ms (1602.9 µs/solve, 1.68x) |

### B. Single Cold-Start Solves

Solving network flow instances from scratch (including initial basis setup and full factorization):

| Instance | Matrix Dimensions | HiGHS C++ CLI | TinyHiGHS.jl | Speedup | Status / Obj Gap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `netflow_small_01.lp` | $88 \times 107$ | 7.73 ms | **0.24 ms** | **32.11x** | `kOptimal`, exact |
| `netflow_small_02.lp` | $88 \times 107$ | 6.76 ms | **0.28 ms** | **23.92x** | `kOptimal`, exact |
| `netflow_medium_02.lp` | $431 \times 764$ | 11.37 ms | **3.23 ms** | **3.52x** | `kOptimal`, exact |

The current comparison script also exercises infeasible and unbounded fixtures;
those rows are deliberately not presented as speedups in this table.

---

## Reproducibility Checklist

- **Correctness is explicit**: the native C++ replay checks equal objectives and
  iteration counts. The three-way Julia benchmark labels each TinyHiGHS result
  `OK`, `DIFF`, or `NONOPT`; a speedup is not presented as a correctness claim.
- **Zero JIT Bias**: Warm-start benchmarks run multiple dry iterations to precompile Julia methods before measuring steady-state resolve latency.
- **Hardware Isolation**: Measurements were repeated with CPU governor pinned to avoid frequency scaling artifacts.
