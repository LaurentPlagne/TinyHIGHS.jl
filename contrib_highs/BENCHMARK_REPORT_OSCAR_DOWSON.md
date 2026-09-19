# Warm-Start Sequence Performance in HiGHS: Yggdrasil JLL vs. Native C++ vs. Zero-Allocation Julia (TinyHiGHS)

**Author**: Laurent Plagne (EDF R&D / LaurentPlagne)  
**Target**: Oscar Dowson (@odow) & the HiGHS / JuMP Team  
**Date**: September 2026  
**Repository & Reproducibility Package**: [https://github.com/LaurentPlagne/TinyHIGHS.jl](https://github.com/LaurentPlagne/TinyHIGHS.jl)

---

## 1. Executive Summary

In the context of dynamic programming and hydro-power scheduling at EDF (solving long sequences of closely related linear programs from a warm-started basis), we profiled the execution time of HiGHS down to the micro-architectural level.

During this investigation, we observed **two unexpected and substantial performance gaps** across different compilation and architectural approaches:

1. **The artifact-to-native gap ($\times 3.98$ in this snapshot)**:
   The official `HiGHS_artifact` binary distributed via Julia takes **21.51 ms** (283.0 µs / solve) on `sequence_small` (76 resolves).
   The local `perf/simd-branchless-pivots` build takes **5.41 ms** (71.2 µs / solve), a **3.98x speedup** in this run using the same public C++ API.

2. **The persistent-workspace gap ($\times 1.42$ additional on `sequence_small`)**:
   The pure Julia port ([TinyHiGHS.jl](https://github.com/LaurentPlagne/TinyHIGHS.jl)), architected with persistent pre-allocated workspaces and **zero heap allocations** (`@allocated == 0`), runs the same 76-solve sequence in **3.80 ms** (50.0 µs / solve) — **5.66x faster than `HiGHS_artifact`** and **1.42x faster than local `HiGHS_branchless`** in this snapshot.

On `sequence_medium` (100 resolves of dimension $1240 \times 1483$), the same
run measured **160.71 ms** for `TinyHiGHS_branchless`, **222.33 ms** for local
`HiGHS_branchless`, and **269.25 ms** for `HiGHS_artifact` (1.68x and 1.38x
relative speedups, respectively).

---

## 2. Experimental Benchmark Results

Measurements performed on **Apple Silicon (macOS aarch64, M-series)** with the
checked-in `bench/bench_3way.jl` (five repetitions for `sequence_small`, three
for `sequence_medium`). Reported values are the best elapsed time from those
repetitions. The benchmark reports objective status explicitly; timings are
machine-dependent snapshots.

### A. Benchmark: `sequence_small` (76 consecutive warm-start resolves, base model $88 \times 107$)

| Engine & Build Configuration | Total Elapsed Time | Avg Time / Solve | Speedup vs artifact | Final Objective / Status |
| :--- | :---: | :---: | :---: | :---: |
| **1. HiGHS_artifact (official v1.15)** | **21.51 ms** | **283.0 µs** | 1.00x *(baseline)* | 7.6042 / n/a |
| **2. HiGHS_branchless (local PR, `-O3`)** | **5.41 ms** | **71.2 µs** | **3.98x** | 7.6042 / OK |
| **3. TinyHiGHS.jl (`kPivotBranching`)** | **4.06 ms** | **53.5 µs** | **5.29x** | 7.6042 / OK |
| **4. TinyHiGHS.jl (`kPivotFdiv`)** | **3.92 ms** | **51.6 µs** | **5.49x** | 7.6042 / OK |
| **5. TinyHiGHS.jl (`kPivotBranchless`)** | **3.80 ms** | **50.0 µs** | **5.66x** | 7.6042 / OK |

---

### B. Benchmark: `sequence_medium` (100 consecutive warm-start resolves, base model $1240 \times 1483$)

| Engine & Build Configuration | Total Elapsed Time | Avg Time / Solve | Speedup vs artifact | Final Objective / Status |
| :--- | :---: | :---: | :---: | :---: |
| **1. HiGHS_artifact (official v1.15)** | **269.25 ms** | **2 692.5 µs** | 1.00x *(baseline)* | -3 717 944.5350 / n/a |
| **2. HiGHS_branchless (local PR, `-O3`)** | **222.33 ms** | **2 223.3 µs** | **1.21x** | -3 717 944.5350 / OK |
| **3. TinyHiGHS.jl (`kPivotBranching`)** | **159.71 ms** | **1 597.1 µs** | **1.69x** | -3 717 944.5347 / OK |
| **4. TinyHiGHS.jl (`kPivotFdiv`)** | **160.29 ms** | **1 602.9 µs** | **1.68x** | -3 717 944.5347 / OK |
| **5. TinyHiGHS.jl (`kPivotBranchless`)** | **160.71 ms** | **1 607.1 µs** | **1.68x** | -3 717 944.5347 / OK |

*Note: Objectives match within the benchmark tolerance; different simplex basis
paths may occur because of tie-breaking in dense pricing.*

---

## 3. Technical Breakdown of the Two Gaps

### Gap 1: Why does local `clang++ -O3` outperform `HiGHS_artifact` by about $4\times$?

In `JuliaPackaging/Yggdrasil`, `HiGHS/build_tarballs.jl` compiles via `BinaryBuilder.jl` in an Alpine Linux cross-compilation environment with:
```cmake
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=${prefix} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
    -DCMAKE_BUILD_TYPE=Release
```

1. **Generic Target Baseline**: To ensure binary compatibility across all systems, BinaryBuilder targets a generic baseline (`armv8.0-a` on AArch64, SSE2 on x86_64). It cannot use `-march=native` or `-mcpu=apple-m1`. On modern micro-architectures (wide-decode pipelines, aggressive out-of-order reordering, specialized vector units), this generic scheduling leaves significant IPC (instructions per cycle) on the table.
2. **Dynamic Linking & Lack of LTO**: Cross-boundary calls into `libhighs.dylib` cannot be inlined or optimized across translation units without Link-Time Optimization (LTO).
3. **Open Question for Yggdrasil**: Could the official HiGHS artifact benefit from targeted optimization flags, LTO (`-flto`), or architecture-specific sub-targets (similar to what is done for OpenBLAS)?

---

### Gap 2: Why does TinyHiGHS.jl outperform native C++ by about $1.4\times$ (and the artifact by up to $5.7\times$)?

On warm-start sequences, the checked-in runner reports the solve count,
iteration count, final objective, and status for every replay. The timing table
above intentionally reports only values produced by `bench/bench_3way.jl`.

In this micro-solve regime (about 50--70 µs per solve in the current
`sequence_small` snapshot):
1. **Persistent-workspace hypothesis**: TinyHiGHS keeps its simplex workspaces
   allocated across resolves; the test suite checks zero allocations on active
   kernels. The benchmark itself measures end-to-end time only, so it does not
   claim a particular percentage of time spent in `malloc`/`free`.
2. **Persistent In-Place Workspaces in Julia**: TinyHiGHS pre-allocates all internal simplex structures (`HVector`, `HFactor`, `DualRHS`, `DualRow`) up to capacity at initialization. Between consecutive solves, arrays are cleared or marked with pointer resets **with zero heap allocations** (`@allocated == 0`).
3. **Full JIT Inlining**: Julia specializes the execution path from model update
   down to hyper-sparse `solveHyper!` for the active host.

---

## 4. Complete Step-by-Step Instructions to Reproduce

We have made the entire benchmarking suite completely standalone and automated.

### Prerequisites
- A Mac (Apple Silicon or Intel) or Linux machine.
- Julia $\ge 1.9$.
- A standard C++11 compiler (`clang++` or `g++`).
- A local build of HiGHS (optional: if omitted, the script automatically tests the official artifact against `TinyHiGHS.jl`).

---

### Method A: Pure C++ Standalone Runner (Zero Julia Code Needed)

A standalone Bash + C++11 runner is provided. It automatically locates the official artifact in `~/.julia/artifacts`, compiles the C++ harness against both `libhighs` versions, and runs the side-by-side benchmark:

```bash
# 1. Clone the repository
git clone https://github.com/LaurentPlagne/TinyHIGHS.jl.git
cd TinyHIGHS.jl

# 2. Run the automated C++ comparison
# The runner auto-detects ../HiGHS/build when it is checked out next to this
# repository. Alternatively set HIGHS_DIR=/path/to/HiGHS or
# HIGHS_BUILD_DIR=/path/to/HiGHS/build.
./contrib_highs/run_bench_cpp.sh
```

**What this script does:**
1. Compiles `contrib_highs/cpp/replay_sequence.cpp` using official C++ API calls (`Highs::readModel`, `Highs::changeColBounds`, `Highs::run`).
2. Links `replay_sequence_original` against the official HiGHS artifact and its matching public headers.
3. Links `replay_sequence` against the selected local/native HiGHS build.
4. Replays `sequence_small` and `sequence_medium` and prints timing, iterations, and objective validation.

---

### Method B: Full 3-Way Benchmark (JLL vs. Native C++ vs. TinyHiGHS)

To run the complete 3-way benchmark that produced the tables in Section 2:

```bash
# In the TinyHIGHS.jl root directory:
julia --project=. bench/bench_3way.jl
```

---

## 5. Artifacts and Test Files Included in the Package

All benchmark assets are tracked directly in the git repository:

1. **C++ Replay Harness**:  
   [`contrib_highs/cpp/replay_sequence.cpp`](https://github.com/LaurentPlagne/TinyHIGHS.jl/blob/main/contrib_highs/cpp/replay_sequence.cpp)  
   140 lines of standard C++11 using only the public `Highs.h` interface.

2. **Automated Bash Runner**:  
   [`contrib_highs/run_bench_cpp.sh`](https://github.com/LaurentPlagne/TinyHIGHS.jl/blob/main/contrib_highs/run_bench_cpp.sh)

3. **3-Way Benchmark Script**:  
   [`bench/bench_3way.jl`](https://github.com/LaurentPlagne/TinyHIGHS.jl/blob/main/bench/bench_3way.jl)

4. **Benchmark Datasets**:
   - `sequence_small`: [`instances/sequences/sequence_small/base.lp`](https://github.com/LaurentPlagne/TinyHIGHS.jl/blob/main/instances/sequences/sequence_small/base.lp) (88 constraints, 107 variables) + [`operations.txt`](https://github.com/LaurentPlagne/TinyHIGHS.jl/blob/main/instances/sequences/sequence_small/operations.txt) (76 sequential bound update & solve steps).
   - `sequence_medium`: [`instances/sequences/sequence_medium/base.lp`](https://github.com/LaurentPlagne/TinyHIGHS.jl/blob/main/instances/sequences/sequence_medium/base.lp) (1240 constraints, 1483 variables) + [`operations.txt`](https://github.com/LaurentPlagne/TinyHIGHS.jl/blob/main/instances/sequences/sequence_medium/operations.txt) (100 sequential steps).

---

## 6. Suggestions for Discussion

1. **For the official artifact & Yggdrasil**:
   Is there potential to introduce architecture-optimized builds or LTO flags in `HiGHS/build_tarballs.jl`? The current snapshot shows a roughly $4\times$ gap between the official artifact and a local build for this micro-solve workload; the command should be rerun on each target machine before drawing a general conclusion.

2. **For HiGHS C++ (Upstream)**:
   A persistent buffer / workspace reuse mode (avoiding vector allocations on successive `highs.run()` calls when the basis and problem dimensions are unchanged) is a possible direction. Its benefit should be measured with the replay harness rather than inferred from a fixed target latency.
