# Upstream Contribution Proposal for HiGHS C++

This document summarizes performance optimizations identified, validated, and benchmarked during the development of **[TinyHiGHS.jl](https://github.com/LaurentPlagne/TinyHIGHS.jl)** (a faithful, zero-allocation pure Julia port of HiGHS's dual and primal revised simplex solvers).

The unit-pivot short-circuit preserves **strict bit-for-bit IEEE-754 equivalence** against frozen HiGHS reference oracles. The optional reciprocal/branchless experiment is exact for unit pivots and is allowed a one-ULP difference for arbitrary pivots.

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

#### B. Warm-Start Sequences (100% Native C++ API Replay, Zero Julia)
Replayed through `contrib_highs/cpp/replay_sequence.cpp` using official `Highs` C++ API:

| Sequence (`instances/sequences/`) | Solves | HiGHS C++ Original | HiGHS C++ Patched | Speedup | Iterations | Final Obj Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sequence_small` | 76 | 16.97 ms (223 µs/solve) | **4.10 ms (54 µs/solve)** | **4.14x (+75.8%)** | 47 vs 47 | **0.0 (exact)** |
| `sequence_medium` | 100 | 182.50 ms (1.82 ms/solve) | **148.27 ms (1.48 ms/solve)** | **1.23x (+18.8%)** | 549 vs 549 | **0.0 (exact)** |

---

## 2. Standalone C++ Replay Benchmark (Zero Julia Required)

A standalone Bash + C++11 runner is provided to replay the benchmarks with no external dependencies:

```bash
cd TinyHIGHS.jl
./contrib_highs/run_bench_cpp.sh
```

This script:
1. Locates a local `HiGHS/build` checkout when present (or accepts
   `HIGHS_DIR`, `HIGHS_BUILD_DIR`, or `HIGHS_INSTALL`), with the Julia artifact
   as a fallback.
2. Compiles `contrib_highs/cpp/replay_sequence.cpp` using
   `clang++ -O3 -std=c++11` against the selected `libhighs`.
3. Runs side-by-side A/B comparison between the official HiGHS artifact and the
   selected build. The official binary is compiled with the artifact's own
   headers to avoid an ABI mismatch.
4. Reports average time per solve in microseconds, simplex iterations, and objective equivalence.

---

## 3. Pull Request Details for `ERGO-Code/HiGHS`

The pull request branch `perf/unit-diagonal-pivots` contains:
1. **Core Kernel Optimization**:
   - `highs/util/HFactor.cpp`: Short-circuit unit pivots in `ftranU`, `btranU`, and `solveHyper`.
2. **Official Catch2 Test Case**:
   - `check/TestSequenceWarmStart.cpp`: Warm-start sequence regression test (240 assertions passed).
   - `check/CMakeLists.txt`: Registered in the `unit_tests` target.
3. **Anonymized Benchmark Assets**:
   - `check/instances/sequence_small_base.lp` (12 KB)
   - `check/instances/sequence_small_operations.txt` (4 KB)

### How to push and open the PR:
```bash
# In your local HiGHS clone:
git remote add myfork https://github.com/LaurentPlagne/HiGHS.git
git push -u myfork perf/unit-diagonal-pivots
```
Then navigate to `https://github.com/ERGO-Code/HiGHS` and click **"Compare & pull request"**.

---

## 4. Future Roadmap: Zero-Allocation Persistent Buffer Architecture

During stochastic optimization sequences, **[TinyHiGHS.jl](https://github.com/LaurentPlagne/TinyHIGHS.jl) reaches 32.4 µs per solve** (compared to 54 µs for patched HiGHS C++ and 223 µs for unpatched HiGHS C++):
- **Root cause of the remaining gap**: In HiGHS C++, calling `highs.run()` or modifying bounds dynamically reallocates and resizes `std::vector` buffers across the `Highs` -> `HEkk` -> `HFactor` hierarchy.
- **Proposed roadmap item for HiGHS 2.x**: Introduce a dedicated "persistent buffer" warm-start mode where workspace memory is allocated once up to capacity and reused in-place across successive resolves with zero heap allocations.
- **Open-source Reference**: The full zero-allocation implementation and reproducibility scripts are available at:
  👉 **[https://github.com/LaurentPlagne/TinyHIGHS.jl](https://github.com/LaurentPlagne/TinyHIGHS.jl)**
