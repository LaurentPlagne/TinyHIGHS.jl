# Quick Start Guide

This guide covers everything you need to start solving and warm-starting linear programs with TinyHiGHS.jl.

---

## Installation

TinyHiGHS is a pure Julia package with no external binary artifacts or system compilers required.

You can install it directly via the Julia package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/LaurentPlagne/TinyHIGHS.jl.git")
```

Or clone it locally for development:

```bash
git clone https://github.com/LaurentPlagne/TinyHIGHS.jl.git
cd TinyHIGHS.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

## Reading and Solving a `.lp` File

The simplest way to use TinyHiGHS is the high-level `solve_lp` entry point:

```julia
using TinyHiGHS

# Read and solve a standard CPLEX format LP file
status, obj, engine = solve_lp("instances/benchmarks/netflow_small_01.lp")

if status == kOptimal
    println("Successfully solved to optimality!")
    println("Optimal Objective: ", obj)
    
    # Extract primal solution values (first num_col entries)
    num_cols = engine.lp.num_col
    primal_solution = engine.info.workValue[1:num_cols]
    println("Primal Variables: ", primal_solution)
else
    println("Solver terminated with status: ", status)
end
```

## MathOptInterface compatibility

TinyHiGHS exposes an optional MathOptInterface (MOI) optimizer for continuous
linear programs. The adapter is loaded automatically when MOI is present in the
active environment, so existing low-level users keep the zero-dependency core:

```julia
using MathOptInterface
const MOI = MathOptInterface
using TinyHiGHS

model = TinyHiGHS.Optimizer()
x = MOI.add_variable(model)
MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
MOI.optimize!(model)

@assert MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
@assert MOI.get(model, MOI.VariablePrimal(), x) ≈ 1.0
```

The adapter supports scalar-affine objectives and constraints, variable bounds,
minimization/maximization, primal and dual values, time limits, and simplex
iteration limits. For high-frequency mutation sequences, use the native
`SimplexEngine` API below to retain its persistent warm-start basis.

---

## Programmatic Model Creation

You can also construct a `SimplexLp` model directly in Julia:

```julia
using TinyHiGHS

# Minimize: 2 x1 + 3 x2
# Subject to:
#   x1 +   x2 >= 1.0  (row 1)
#   x1 + 2 x2 <= 4.0  (row 2)
# Bounds:
#   0 <= x1 <= 10
#   0 <= x2 <= 10

num_col = 2
num_row = 2

# Constraint matrix A in 1-based Compressed Sparse Column (CSC) format
# Column 1 entries: row 1 (1.0), row 2 (1.0)
# Column 2 entries: row 1 (1.0), row 2 (2.0)
a_start = [1, 3, 5]
a_index = [1, 2, 1, 2]
a_value = [1.0, 1.0, 1.0, 2.0]
a_matrix = SparseMatrix(num_col, num_row, a_start, a_index, a_value)

col_cost  = [2.0, 3.0]
col_lower = [0.0, 0.0]
col_upper = [10.0, 10.0]

row_lower = [1.0, -kHighsInf]
row_upper = [kHighsInf, 4.0]

lp = SimplexLp(num_col, num_row, a_matrix, col_cost, col_lower, col_upper, row_lower, row_upper; sense=kMinimize)

# Solve
status, obj, engine = solve_lp(lp)
println("Status: ", status)
println("Optimal value: ", obj)
```

---

## High-Performance Warm-Starting

The true power of TinyHiGHS lies in iterative workflows where problem parameters change sequentially:

```julia
using TinyHiGHS

# 1. Initialize persistent simplex engine once
engine = SimplexEngine(lp)

# 2. First cold solve
status = solve!(engine)
println("Cold solve: obj = ", engine.info.primal_objective_value)

# 3. Modify bounds or costs in-place
# Change lower and upper bounds of column 1
change_col_bounds!(engine, 1, 0.5, 5.0)

# Change bounds of row 2
change_row_bounds!(engine, 2, -kHighsInf, 3.5)

# 4. Resolve in-place!
# Reuses the exact basis and LU factorization without any memory allocation:
status = solve!(engine)
println("Warm solve: obj = ", engine.info.primal_objective_value)
```

Internal buffers are retained across warm resolves. The timing benchmark and the
allocation probe in `bench/generate_synthetic_sequences.jl` report the observed
behavior; measure `@allocated solve!(engine)` on the Julia version and workload
you deploy before relying on a hard allocation budget.
