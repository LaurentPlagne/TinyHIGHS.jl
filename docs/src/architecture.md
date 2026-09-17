# System Architecture

TinyHiGHS is organized into modular layers designed for maximal memory efficiency, cache locality, and numerical stability.

---

## High-Level Architecture Diagram

The diagram below illustrates how data and control flow through TinyHiGHS:

```mermaid
classDiagram
    class SimplexLp {
        +Int num_col
        +Int num_row
        +SparseMatrix a_matrix
        +Vector~Float64~ col_cost
        +Vector~Float64~ col_lower, col_upper
        +Vector~Float64~ row_lower, row_upper
    }

    class SimplexEngine {
        +SimplexLp lp
        +SimplexBasis basis
        +SimplexInfo info
        +Nla nla
        +SimplexStatus status
        +solve!()
        +change_col_bounds!()
        +change_row_bounds!()
    }

    class HFactor {
        +Int num_row
        +Vector~Int~ basic_index
        +build!()
        +ftranCall!()
        +btranCall!()
        +update!()
    }

    class HVector {
        +Int size
        +Int count
        +Vector~Int~ index
        +Vector~Float64~ array
        +tight!()
        +pack!()
        +saxpy!()
    }

    class DualSolver {
        +HVector row_ep
        +HVector row_ap
        +HVector col_aq
        +HVector col_BFRT
        +solvePhase1!()
        +solvePhase2!()
    }

    class PrimalSolver {
        +HVector row_ep
        +HVector row_ap
        +HVector col_aq
        +solvePhase1!()
        +solvePhase2!()
    }

    SimplexEngine --> SimplexLp : holds
    SimplexEngine --> HFactor : basis factorization
    SimplexEngine --> DualSolver : dual iterations
    SimplexEngine --> PrimalSolver : primal iterations
    DualSolver --> HVector : sparse vectors
    PrimalSolver --> HVector : sparse vectors
```

---

## Revised Simplex Iteration Lifecycle

In the dual simplex method (default), each iteration performs a pivot to drive primal infeasibilities to zero while maintaining dual feasibility.

```mermaid
sequenceDiagram
    autonumber
    participant D as DualSolver
    participant E as SimplexEngine
    participant H as HFactor (LU)
    participant V as HVector Workspaces

    Note over D: Step 1: Ratio Test & Row Selection
    D->>D: chooseRow (DSE or Devex pricing)
    Note right of D: Identifies leaving variable (row_out)

    Note over D,H: Step 2: Backward Transformation (BTRAN)
    D->>H: btranCall!(row_ep) where row_ep = e_{row_out}
    H-->>D: π = B^{-T} e_{row_out}

    Note over D,V: Step 3: Tableau Row Pricing
    D->>V: tableau_row_price!(row_ap = π^T A)
    
    Note over D: Step 4: Column Selection
    D->>D: chooseColumn via Bound-Flipping Ratio Test (BFRT)
    Note right of D: Identifies entering variable (variable_in)

    Note over D,H: Step 5: Forward Transformation (FTRAN)
    D->>H: ftranCall!(col_aq) where col_aq = A_{*, variable_in}
    H-->>D: α = B^{-1} A_{*, variable_in}

    Note over D,H: Step 6: Basis & Factorization Update
    D->>H: update!(col_aq, row_ep, row_out)
    Note right of H: Forrest-Tomlin update in-place
    D->>E: update_pivots!(variable_in, row_out)
```

---

## Core Components

### 1. `SimplexEngine`
The central state container. It owns the problem specification, basis history, control flags, and preallocated working memory. All algorithmic actions (dual iterations, primal iterations, scaling, perturbation, bound modifications) operate on this engine.

### 2. `HVector` (Sparse/Dense Workspaces)
HiGHS and TinyHiGHS rely on a hybrid sparse/dense vector structure:
- A dense array of length $m$ provides $O(1)$ lookup.
- A non-zero index list `index[1:count]` enables $O(\text{nnz})$ sparse traversal and hyper-sparse graph walks.
- When fill-in exceeds ~30%, routines automatically switch to dense SIMD loops to maximize hardware memory bandwidth.

### 3. `HFactor` (LU Basis Factorization)
Maintains the factorization $P B Q = L U$:
- **Initial Factorization (`build!`)**: Uses Markowitz threshold pivoting to minimize fill-in.
- **Basis Updates (`update!`)**: Applies Forrest-Tomlin updates to maintain $L$ and $U$ without full refactorization.
- **Unit Pivot Bypass**: Directly executes $x / (\pm 1.0)$ as identity or negation, avoiding hardware `FDIV` stalls.

### 4. `DualSolver` & `PrimalSolver`
- **Dual Steepest Edge (DSE)**: Provides steep progress directions with adaptive switching to Devex if weight computation becomes expensive.
- **Bound-Flipping Ratio Test (BFRT)**: Allows passing multiple non-basic bounds in a single iteration, dramatically cutting down iteration counts on box-constrained models.
- **Cycling Mitigation**: 64-bit basis hash tracking and taboo lists prevent degeneracy loops.
