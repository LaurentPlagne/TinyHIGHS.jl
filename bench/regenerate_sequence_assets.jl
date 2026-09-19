"""Regenerate the checked-in warm-start sequence LP snapshots.

The operation logs use one-based column and row indices.  The base model is
rewritten before replay so the canonical `c0`, `c1`, ... order is explicit in
the objective and survives a round trip through an LP reader.
"""

using TinyHiGHS
using Printf

const SEQUENCE_ROOT = normpath(joinpath(@__DIR__, "..", "instances", "sequences"))

function apply_operation!(engine::SimplexEngine, tokens::Vector{<:AbstractString})::Bool
    isempty(tokens) && return false
    op = tokens[1]
    if op == "change_col_bounds"
        change_col_bounds!(engine, parse(Int, tokens[2]),
            parse(Float64, tokens[3]), parse(Float64, tokens[4]))
    elseif op == "change_cols_bounds"
        n = parse(Int, tokens[2])
        cols = Vector{Int}(undef, n)
        lowers = Vector{Float64}(undef, n)
        uppers = Vector{Float64}(undef, n)
        position = 3
        for k in 1:n
            cols[k] = parse(Int, tokens[position])
            lowers[k] = parse(Float64, tokens[position + 1])
            uppers[k] = parse(Float64, tokens[position + 2])
            position += 3
        end
        change_cols_bounds!(engine, cols, lowers, uppers)
    elseif op == "change_row_bounds"
        change_row_bounds!(engine, parse(Int, tokens[2]),
            parse(Float64, tokens[3]), parse(Float64, tokens[4]))
    elseif op == "change_rows_bounds"
        n = parse(Int, tokens[2])
        rows = Vector{Int}(undef, n)
        lowers = Vector{Float64}(undef, n)
        uppers = Vector{Float64}(undef, n)
        position = 3
        for k in 1:n
            rows[k] = parse(Int, tokens[position])
            lowers[k] = parse(Float64, tokens[position + 1])
            uppers[k] = parse(Float64, tokens[position + 2])
            position += 3
        end
        change_rows_bounds!(engine, rows, lowers, uppers)
    elseif op == "change_cols_cost"
        n = parse(Int, tokens[2])
        cols = Vector{Int}(undef, n)
        costs = Vector{Float64}(undef, n)
        position = 3
        for k in 1:n
            cols[k] = parse(Int, tokens[position])
            costs[k] = parse(Float64, tokens[position + 1])
            position += 2
        end
        change_cols_cost!(engine, cols, costs)
    elseif op == "solve"
        status = solve!(engine)
        status == TinyHiGHS.kOptimal || error("sequence solve returned $status")
        return true
    elseif op == "end"
        return false
    else
        error("unknown sequence operation: $op")
    end
    return false
end

function regenerate_sequence!(name::AbstractString)
    root = joinpath(SEQUENCE_ROOT, name)
    base_path = joinpath(root, "base.lp")
    operations_path = joinpath(root, "operations.txt")
    steps_dir = joinpath(root, "steps")
    mkpath(steps_dir)

    # Rewriting the base is intentional: it materializes the canonical column
    # order in the objective before any numeric replay operation is applied.
    lp = read_lp(base_path)
    write_lp(base_path, lp)
    engine = SimplexEngine(lp)

    solve_count = 0
    for line in eachline(operations_path)
        solved = apply_operation!(engine, split(strip(line)))
        solved || continue
        solve_count += 1
        solve_count <= 10 || continue
        write_lp(joinpath(steps_dir, @sprintf("step_%03d.lp", solve_count)), engine.lp)
    end
    @info "regenerated sequence" name solve_count
end

for name in ("sequence_small", "sequence_medium")
    regenerate_sequence!(name)
end
