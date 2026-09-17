# API Reference

This section provides the complete reference documentation for TinyHiGHS.jl.

---

## High-Level API

```@docs
solve_lp
read_lp
write_lp
```

---

## Simplex Engine & In-Place Modification

```@docs
SimplexEngine
SimplexLp
SimplexBasis
SimplexOptions
SimplexInfo
solve!
change_col_bounds!
change_cols_bounds!
change_row_bounds!
change_rows_bounds!
change_cols_cost!
```

---

## Linear Algebra & Factorization

```@docs
HFactor
build!
Nla
ftran!
btran!
```

---

## Sparse Data Structures

```@docs
HVector
setup!(::HVector, ::Int)
clear!(::HVector)
tight!(::HVector)
pack!(::HVector)
saxpy!(::HVector, ::Float64, ::HVector)
norm2(::HVector)
SparseMatrix
```

---

## Solvers

```@docs
DualSolver
PrimalSolver
```
