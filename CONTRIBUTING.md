# Contributing to TinyHiGHS.jl

TinyHiGHS.jl is an experimental numerical solver and benchmark suite. Small,
focused pull requests are welcome, especially when they include a reproducible
test or benchmark and explain any numerical tolerance used.

## Development setup

The package targets Julia 1.10 and newer. After cloning the repository:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

The repository's CI exercises the full test matrix. The optional MathOptInterface
adapter is covered by the test target and should remain compatible with the
declared MOI version range.

## Benchmarks and documentation

Run the three-way warm-start benchmark with:

```bash
julia --project=. bench/bench_3way.jl
```

The C++ replay comparison is available through
`contrib_highs/run_bench_cpp.sh`. Timings are machine-dependent; include the
printed Julia version, CPU, compiler, and commit when reporting a result. Build
the documentation locally with `julia --project=docs docs/make.jl`.

Please do not commit generated build directories, local binaries, manifests, or
private benchmark data. See `.gitignore` for the repository defaults.

## Pull requests

Describe the motivation, the validation performed, and any known numerical or
performance trade-off. Do not include confidential source data in issues or
pull requests; use the public fixtures under `instances/` or a minimal synthetic
example instead.
