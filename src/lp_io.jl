# ==============================================================================
# Native CPLEX .lp file reader and writer
# Zero external dependencies — pure Julia stdlib
# ==============================================================================

"""
    write_lp(filename_or_io, lp::SimplexLp; var_names=nothing, row_names=nothing)

Write a `SimplexLp` instance to standard CPLEX `.lp` format (plain text).

# Arguments
- `filename_or_io`: Either an `IO` stream or a `String` containing the destination file path.
- `lp::SimplexLp`: The linear programming model to export. Constraint matrix `lp.a_matrix` must be in column-wise format.
- `var_names`: Optional `Vector{String}` of variable names (length `num_col`). Defaults to `c0`, `c1`, ...
- `row_names`: Optional `Vector{String}` of constraint names (length `num_row`). Defaults to `r0`, `r1`, ...

# Example
```julia
using TinyHiGHS

lp = read_lp("model.lp")
write_lp("exported.lp", lp)
```
"""
function write_lp(io::IO, lp::SimplexLp; var_names=nothing, row_names=nothing)
    m = lp.a_matrix
    is_colwise(m) || error("write_lp requires matrix A to be in column-wise format")

    vnames = if isnothing(var_names)
        ["c$(j-1)" for j in 1:lp.num_col]
    else
        var_names
    end
    rnames = if isnothing(row_names)
        ["r$(i-1)" for i in 1:lp.num_row]
    else
        row_names
    end

    println(io, "\\ Problem written by TinyHiGHS")
    println(io, lp.sense == kMaximize ? "Maximize" : "Minimize")
    print(io, " obj:")
    wrote_obj = false
    # Emit zero-cost columns as explicit zero terms as well.  LP readers
    # discover variables in textual order; listing every column here makes the
    # default c0..cN names a stable positional schema for numeric replay logs.
    for j in 1:lp.num_col
        c = lp.col_cost[j]
        sgn = c >= 0 ? "+" : "-"
        val = abs(c)
        print(io, " ", sgn, val, " ", vnames[j])
        wrote_obj = true
    end
    wrote_obj || print(io, " 0")
    println(io)

    println(io, "Subject To")
    # Build row terms for readable display
    rows_terms = [Tuple{Int,Float64}[] for _ in 1:lp.num_row]
    for j in 1:lp.num_col
        for p in m.start[j]:(m.start[j+1] - 1)
            push!(rows_terms[m.index[p]], (j, m.value[p]))
        end
    end

    for i in 1:lp.num_row
        print(io, " ", rnames[i], ":")
        terms = rows_terms[i]
        if isempty(terms)
            print(io, " 0")
        else
            for (j, coeff) in terms
                sgn = coeff >= 0 ? "+" : "-"
                print(io, " ", sgn, abs(coeff), " ", vnames[j])
            end
        end
        rl, ru = lp.row_lower[i], lp.row_upper[i]
        if rl == ru
            println(io, " = ", rl)
        elseif ru == kHighsInf
            println(io, " >= ", rl)
        elseif rl == -kHighsInf
            println(io, " <= ", ru)
        else
            # Ranged constraint: decomposed into upper bound
            # as standard .lp format does not natively support two-sided bounds in Subject To
            println(io, " <= ", ru)
        end
    end

    println(io, "Bounds")
    for j in 1:lp.num_col
        cl, cu = lp.col_lower[j], lp.col_upper[j]
        vname = vnames[j]
        if cl == -kHighsInf && cu == kHighsInf
            println(io, " ", vname, " free")
        elseif cl == -kHighsInf
            println(io, " -inf <= ", vname, " <= ", cu)
        elseif cu == kHighsInf
            println(io, " ", vname, " >= ", cl)
        elseif cl == cu
            println(io, " ", vname, " = ", cl)
        else
            println(io, " ", cl, " <= ", vname, " <= ", cu)
        end
    end
    println(io, "End")
    return nothing
end

function write_lp(filename::AbstractString, lp::SimplexLp; kwargs...)
    open(filename, "w") do io
        write_lp(io, lp; kwargs...)
    end
end

"""
    read_lp(filename_or_io) -> SimplexLp

Parse a linear program from a CPLEX `.lp` file or `IO` stream into a `SimplexLp`.

This native parser is implemented in 100% pure Julia stdlib without any external C/C++
dependencies. It supports:
- Objective sense: `Minimize` / `Maximize` (and abbreviations `min`, `max`)
- Linear combinations of terms with arbitrary signs and floating-point coefficients
- Constraints: `Subject To`, `st`, `s.t.`, `such that` with `<=`, `>=`, `=` relations
- Variable bounds: `Bounds` section with lower bounds, upper bounds, equality, and `free` variables
- Single-line and multi-line comments starting with `\\`

# Arguments
- `filename_or_io`: A file path (`AbstractString`) or an open `IO` stream.

# Returns
- A `SimplexLp` model ready to be solved or inspected.

# Example
```julia
using TinyHiGHS

lp = read_lp("instance.lp")
println("Columns: ", lp.num_col, ", Rows: ", lp.num_row)
```
"""
function read_lp(filename_or_io)::SimplexLp
    lines = if filename_or_io isa IO
        readlines(filename_or_io)
    else
        readlines(filename_or_io)
    end

    # Section splitting
    section = :none
    sense = kMinimize
    obj_name = "obj"
    obj_tokens = SubString{String}[]
    constraint_lines = Tuple{String, Vector{SubString{String}}}[]
    bound_lines = SubString{String}[]

    for line in lines
        # Strip comments
        c_idx = findfirst(==('\\'), line)
        active_part = isnothing(c_idx) ? strip(line) : strip(line[1:prevind(line, c_idx)])
        isempty(active_part) && continue

        low = lowercase(active_part)
        if low in ("min", "minimize")
            section = :obj
            sense = kMinimize
            continue
        elseif low in ("max", "maximize")
            section = :obj
            sense = kMaximize
            continue
        elseif low in ("subject to", "such that", "st", "s.t.")
            section = :constraints
            continue
        elseif low in ("bounds", "bound")
            section = :bounds
            continue
        elseif low in ("end",)
            break
        end

        if section == :obj
            append!(obj_tokens, Base.split(active_part))
        elseif section == :constraints
            # New constraint or continuation line
            push!(constraint_lines, (active_part, Base.split(active_part)))
        elseif section == :bounds
            push!(bound_lines, SubString(active_part, 1))
        end
    end

    # Variable lookup table
    var2col = Dict{String,Int}()
    col_names = String[]
    function get_col_id!(v::AbstractString)
        str = String(v)
        get!(var2col, str) do
            push!(col_names, str)
            length(col_names)
        end
    end

    # Linear polynomial parser: [ "+", "1.5", "x1", "-", "x2" ] or [ "+1.5x1", "-x2" ]
    function parse_linear_terms(tokens)
        terms = Tuple{Float64, String}[]
        sign = 1.0
        pending_coeff = nothing
        i = 1
        while i <= length(tokens)
            tok = tokens[i]
            if tok == "+"
                sign = 1.0
                i += 1
                continue
            elseif tok == "-"
                sign = -1.0
                i += 1
                continue
            end

            # Check if token starts with + or -
            if startswith(tok, "+")
                sign = 1.0
                tok = tok[2:end]
            elseif startswith(tok, "-")
                sign = -1.0
                tok = tok[2:end]
            end

            # If a coefficient is pending, the current token is the variable name
            # (even if the variable name is numeric, e.g., in Netlib instances written by HiGHS)
            if !isnothing(pending_coeff)
                push!(terms, (pending_coeff, String(tok)))
                pending_coeff = nothing
                sign = 1.0
                i += 1
                continue
            end

            # Is it a pure numeric value?
            val = tryparse(Float64, tok)
            if !isnothing(val)
                pending_coeff = sign * val
                sign = 1.0
                i += 1
                continue
            end

            # Variable name without explicit coefficient (implicit 1.0 or -1.0)
            push!(terms, (sign, String(tok)))
            sign = 1.0
            i += 1
        end
        return terms
    end

    # Parse objective
    # Optional leading objective name "obj:"
    if !isempty(obj_tokens) && endswith(obj_tokens[1], ":")
        popfirst!(obj_tokens)
    end
    obj_terms = parse_linear_terms(obj_tokens)
    for (_, vname) in obj_terms
        get_col_id!(vname)
    end

    # Parse constraints
    raw_constraints = @NamedTuple{name::String, lhs_tokens::Vector{SubString{String}}, op::String, rhs::Float64}[]
    current_name = ""
    current_tokens = SubString{String}[]

    for (raw_str, toks) in constraint_lines
        t_first = toks[1]
        c_colon = findfirst(==(':'), raw_str)
        if !isnothing(c_colon)
            # Close previous constraint if existing
            if !isempty(current_tokens)
                # Find operator and RHS
                op_idx = findfirst(t -> t in ("<=", ">=", "=", "<", ">"), current_tokens)
                if !isnothing(op_idx)
                    op = String(current_tokens[op_idx])
                    rhs = parse(Float64, current_tokens[op_idx+1])
                    push!(raw_constraints, (name=current_name, lhs_tokens=current_tokens[1:op_idx-1], op=op, rhs=rhs))
                end
                empty!(current_tokens)
            end
            c_name = strip(raw_str[1:prevind(raw_str, c_colon)])
            current_name = String(c_name)
            rest_toks = Base.split(raw_str[nextind(raw_str, c_colon):end])
            append!(current_tokens, rest_toks)
        else
            append!(current_tokens, toks)
        end
    end
    if !isempty(current_tokens)
        op_idx = findfirst(t -> t in ("<=", ">=", "=", "<", ">"), current_tokens)
        if !isnothing(op_idx)
            op = String(current_tokens[op_idx])
            rhs = parse(Float64, current_tokens[op_idx+1])
            push!(raw_constraints, (name=current_name, lhs_tokens=current_tokens[1:op_idx-1], op=op, rhs=rhs))
        end
    end

    # Register all constraint variables
    parsed_lhs = Vector{Tuple{Float64, String}}[]
    for c in raw_constraints
        terms = parse_linear_terms(c.lhs_tokens)
        for (_, vname) in terms
            get_col_id!(vname)
        end
        push!(parsed_lhs, terms)
    end

    num_col = length(col_names)
    num_row = length(raw_constraints)

    # Column costs
    col_cost = zeros(Float64, num_col)
    for (coeff, vname) in obj_terms
        col_cost[var2col[vname]] += coeff
    end

    # Row bounds
    row_lower = fill(-kHighsInf, num_row)
    row_upper = fill(kHighsInf, num_row)
    for (i, c) in enumerate(raw_constraints)
        if c.op in ("<=", "<")
            row_upper[i] = c.rhs
        elseif c.op in (">=", ">")
            row_lower[i] = c.rhs
        elseif c.op == "="
            row_lower[i] = c.rhs
            row_upper[i] = c.rhs
        end
    end

    # Keep the parser's first-occurrence order while collecting the model.
    # It is reordered below once bounds have been parsed.  Files written by
    # `write_lp` use the stable synthetic names `c0`, `c1`, ..., so their
    # numeric order is the only ordering that survives an LP round trip.
    col_entries = [Tuple{Int,Float64}[] for _ in 1:num_col]
    for (iRow, terms) in enumerate(parsed_lhs)
        for (coeff, vname) in terms
            jCol = var2col[vname]
            push!(col_entries[jCol], (iRow, coeff))
        end
    end

    # Variable bounds: default [0, Inf) in standard CPLEX LP
    col_lower = zeros(Float64, num_col)
    col_upper = fill(kHighsInf, num_col)

    for b_line in bound_lines
        toks = Base.split(b_line)
        isempty(toks) && continue
        # Case 1: "var free"
        if length(toks) >= 2 && lowercase(toks[2]) == "free"
            vname = String(toks[1])
            if haskey(var2col, vname)
                j = var2col[vname]
                col_lower[j] = -kHighsInf
                col_upper[j] = kHighsInf
            end
            continue
        end
        # Case 2: "l <= var <= u"
        if length(toks) == 5 && toks[2] == "<=" && toks[4] == "<="
            l_val = parse_bound_val(toks[1])
            vname = String(toks[3])
            u_val = parse_bound_val(toks[5])
            if haskey(var2col, vname)
                j = var2col[vname]
                col_lower[j] = l_val
                col_upper[j] = u_val
            end
            continue
        end
        # Case 3: "var <= u" or "var >= l" or "var = val"
        if length(toks) == 3
            vname = String(toks[1])
            op = toks[2]
            val = parse_bound_val(toks[3])
            if haskey(var2col, vname)
                j = var2col[vname]
                if op in ("<=", "<")
                    col_upper[j] = val
                elseif op in (">=", ">")
                    col_lower[j] = val
                elseif op == "="
                    col_lower[j] = val
                    col_upper[j] = val
                end
            end
            continue
        end
        # Case 4: "l <= var"
        if length(toks) == 3 && toks[2] == "<="
            val = parse_bound_val(toks[1])
            vname = String(toks[3])
            if haskey(var2col, vname)
                j = var2col[vname]
                col_lower[j] = val
            end
            continue
        end
    end

    # `write_lp` deliberately gives anonymized columns the names c0..cN.  A
    # standard LP reader is free to discover variables in textual order, which
    # puts objective-only columns before columns first seen in constraints.  In
    # that case a numeric operation such as `change_col_bounds 1 ...` would
    # silently target the wrong column after a round trip.  Recover the stable
    # order for this canonical name family; arbitrary user names retain the
    # normal first-occurrence order.
    column_order = collect(1:num_col)
    canonical = [get(var2col, "c$(j - 1)", 0) for j in 1:num_col]
    if all(!iszero, canonical) && length(unique(canonical)) == num_col
        column_order = canonical
    end

    ordered_col_cost = col_cost[column_order]
    ordered_col_lower = col_lower[column_order]
    ordered_col_upper = col_upper[column_order]

    # CSC sparse matrix in the stable column order, with rows sorted.
    a_start = zeros(Int, num_col + 1)
    a_start[1] = 1
    total_nz = sum(length, col_entries; init=0)
    a_index = zeros(Int, total_nz)
    a_value = zeros(Float64, total_nz)

    pos = 1
    for j in 1:num_col
        old_j = column_order[j]
        sort!(col_entries[old_j], by=first)
        for (iRow, val) in col_entries[old_j]
            a_index[pos] = iRow
            a_value[pos] = val
            pos += 1
        end
        a_start[j + 1] = pos
    end
    matrix = SparseMatrix(num_col, num_row, a_start, a_index, a_value)

    return SimplexLp(num_col, num_row, matrix, ordered_col_cost,
        ordered_col_lower, ordered_col_upper, row_lower, row_upper; sense=sense)
end

function parse_bound_val(s::AbstractString)::Float64
    ls = lowercase(s)
    ls in ("+inf", "inf") && return kHighsInf
    ls in ("-inf",) && return -kHighsInf
    return parse(Float64, s)
end

# ==============================================================================
# High-level API: solve! and solve_lp
# ==============================================================================

"""
    solve!(engine::SimplexEngine; algorithm::SimplexAlgorithm=kDual) -> ModelStatus

Solve the linear program currently loaded into `engine`.

If the engine state is fresh or has been modified (bounds, costs, or matrix entries),
it automatically initialises the basis, solves phases 1 and 2, and performs required
rebuilds and refactorizations in-place.

# Arguments
- `engine::SimplexEngine`: The persistent simplex state and preallocated workspace buffers.
- `algorithm::SimplexAlgorithm`: Optimization algorithm to use, either `kDual` (dual simplex, default) or `kPrimal` (primal simplex).

# Returns
- `ModelStatus`: Optimization outcome (e.g., `kOptimal`, `kInfeasible`, `kUnbounded`).

# Performance Note
When reusing the same `engine` across a sequence of resolves with modified bounds or costs
(`change_col_bounds!`, `change_row_bounds!`, `change_cols_cost!`), this method achieves
**zero heap allocations** and microsecond warm-start resolution times.
"""
function solve!(engine::SimplexEngine; algorithm=kDual)
    initialise_for_solve!(engine)
    solver = algorithm == kDual ? DualSolver(engine) : PrimalSolver(engine)
    solve!(solver)
    return engine.model_status
end

"""
    solve_lp(lp_or_path; algorithm::SimplexAlgorithm=kDual) -> (status, objective_value, engine)

Convenience high-level entry point to load (if path) and solve a linear program.

# Arguments
- `lp_or_path`: Either a `SimplexLp` instance or a `String` representing a path to a `.lp` file.
- `algorithm::SimplexAlgorithm`: Solver algorithm (`kDual` or `kPrimal`, defaults to `kDual`).

# Returns
A 3-tuple `(status, objective_value, engine)`:
- `status::ModelStatus`: Final model status (e.g., `kOptimal`).
- `objective_value::Float64`: Primal objective value at termination.
- `engine::SimplexEngine`: The solved simplex engine instance, allowing extraction of primal/dual solution vectors and basis information.

# Example
```julia
using TinyHiGHS

status, obj, engine = solve_lp("instances/benchmarks/netflow_small_01.lp")
if status == kOptimal
    println("Optimal objective: ", obj)
    println("Primal values: ", engine.info.workValue[1:engine.lp.num_col])
end
```
"""
function solve_lp(lp::SimplexLp; algorithm=kDual)
    engine = SimplexEngine(lp)
    status = solve!(engine; algorithm=algorithm)
    return status, engine.info.primal_objective_value, engine
end

solve_lp(path::AbstractString; kwargs...) = solve_lp(read_lp(path); kwargs...)
