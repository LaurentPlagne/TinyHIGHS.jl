# HiGHS experimental patches

`0001-hfactor-short-circuit-unit-pivots.patch` is a historical baseline used to
measure the unit-pivot special case. It is retained for provenance and is not
the current branchless experiment.

The current reciprocal/SIMD experiment is represented by
`0002-hfactor-simd-branchless-pivot-inverses.patch` and the upstream branch
`perf/simd-branchless-pivots`. Apply patches only to the matching HiGHS source
revision described in the accompanying benchmark notes.
