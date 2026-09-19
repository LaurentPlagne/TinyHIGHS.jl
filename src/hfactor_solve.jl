# Factor solves: HFactor.cpp:1529-1915 and solveHyper:61.

const EMPTY_FLOAT64_VEC = Float64[]

# Active strategy for U pivot inversion
const ACTIVE_PIVOT_STRATEGY = Ref{PivotStrategy}(kPivotBranching)

"""
    set_pivot_strategy!(s::PivotStrategy)

Sets the global pivot inversion strategy in HFactor:
- `kPivotBranching`: branching reference path (bit-exact with HiGHS)
- `kPivotBranchless`: branchless multiplication by a pre-computed reciprocal
  (bit-exact for `±1` and powers of two; arbitrary pivots may differ by one ULP)
- `kPivotFdiv`: unconditional floating-point division (original HiGHS)
"""
function set_pivot_strategy!(s::PivotStrategy)
    ACTIVE_PIVOT_STRATEGY[] = s
    return s
end

@inline function apply_pivot(::Val{kPivotBranching}, pivot_multiplier::Float64, pivot_val::Float64, pivot_inv::Float64)
    if pivot_val != 1.0
        return pivot_val == -1.0 ? -pivot_multiplier : pivot_multiplier / pivot_val
    end
    return pivot_multiplier
end

@inline function apply_pivot(::Val{kPivotBranchless}, pivot_multiplier::Float64, pivot_val::Float64, pivot_inv::Float64)
    return pivot_multiplier * pivot_inv
end

@inline function apply_pivot(::Val{kPivotFdiv}, pivot_multiplier::Float64, pivot_val::Float64, pivot_inv::Float64)
    return pivot_multiplier / pivot_val
end

"""
    solveHyper!(rhs, h_size, h_lookup, h_pivot_index, h_pivot_value,
                has_pivot_value, h_start, h_end, end_shift, h_index, h_value,
                h_pivot_inv_value=EMPTY_FLOAT64_VEC, strategy=Val(ACTIVE_PIVOT_STRATEGY[]))

`HFactor::solveHyper`: constructs hyper-sparse list via depth-first search,
then applies solves in reverse order. `h_end[i + end_shift]` gives the end
of column `i` (`end_shift = 1` for L, `0` for U).
"""
function solveHyper!(rhs::HVector, h_size::Int, h_lookup::Vector{Int},
    h_pivot_index::Vector{Int}, h_pivot_value::Vector{Float64},
    has_pivot_value::Bool, h_start::Vector{Int}, h_end::Vector{Int},
    end_shift::Int, h_index::Vector{Int}, h_value::Vector{Float64},
    h_pivot_inv_value::Vector{Float64}=EMPTY_FLOAT64_VEC,
    strategy::Val{S}=Val(ACTIVE_PIVOT_STRATEGY[])) where S
    rhs_count = rhs.count
    list_mark = rhs.cwork
    list_index = rhs.iwork
    stack_base = h_size
    list_count = 0
    count_pivot = 0
    count_entry = 0

    @inbounds for i ∈ 1:rhs_count
        i_trans = h_lookup[rhs.index[i]]
        list_mark[i_trans] == 0x01 && continue
        Hi = i_trans
        Hk = h_start[Hi]
        n_stack = -1
        list_mark[Hi] = 0x01
        while true
            if Hk < h_end[Hi + end_shift]
                Hi_sub = h_lookup[h_index[Hk]]
                Hk += 1
                if list_mark[Hi_sub] == 0x00
                    list_mark[Hi_sub] = 0x01
                    n_stack += 1
                    list_index[stack_base + n_stack + 1] = Hi
                    n_stack += 1
                    list_index[stack_base + n_stack + 1] = Hk
                    Hi = Hi_sub
                    Hk = h_start[Hi]
                    if Hi >= h_size
                        count_pivot += 1
                        count_entry += h_end[Hi + end_shift] - h_start[Hi]
                    end
                end
            else
                list_count += 1
                list_index[list_count] = Hi
                n_stack == -1 && break
                Hk = list_index[stack_base + n_stack + 1]
                n_stack -= 1
                Hi = list_index[stack_base + n_stack + 1]
                n_stack -= 1
            end
        end
    end
    rhs.synthetic_tick += count_pivot * 20 + count_entry * 10

    if !has_pivot_value
        rhs_count = 0
        @inbounds for iList ∈ list_count:-1:1
            i = list_index[iList]
            list_mark[i] = 0x00
            pivotRow = h_pivot_index[i]
            pivot_multiplier = rhs.array[pivotRow]
            if abs(pivot_multiplier) > kHighsTiny
                rhs_count += 1
                rhs.index[rhs_count] = pivotRow
                for k ∈ h_start[i]:(h_end[i + end_shift] - 1)
                    rhs.array[h_index[k]] -= pivot_multiplier * h_value[k]
                end
            else
                rhs.array[pivotRow] = 0.0
            end
        end
        rhs.count = rhs_count
    else
        rhs_count = 0
        has_inv = !isempty(h_pivot_inv_value)
        @inbounds for iList ∈ list_count:-1:1
            i = list_index[iList]
            list_mark[i] = 0x00
            pivotRow = h_pivot_index[i]
            pivot_multiplier = rhs.array[pivotRow]
            if abs(pivot_multiplier) > kHighsTiny
                pivot_val = h_pivot_value[i]
                pivot_inv = has_inv ? h_pivot_inv_value[i] : 1.0 / pivot_val
                pivot_multiplier = apply_pivot(strategy, pivot_multiplier, pivot_val, pivot_inv)
                rhs.array[pivotRow] = pivot_multiplier
                rhs_count += 1
                rhs.index[rhs_count] = pivotRow
                for k ∈ h_start[i]:(h_end[i + end_shift] - 1)
                    rhs.array[h_index[k]] -= pivot_multiplier * h_value[k]
                end
            else
                rhs.array[pivotRow] = 0.0
            end
        end
        rhs.count = rhs_count
    end
    return rhs
end

"""`HFactor::ftranFT` — Forrest-Tomlin update solve (empty without update)."""
function ftranFT!(f::HFactor, vector::HVector)
    rhs_count = vector.count
    pf_pivot_count = length(f.pf_pivot_index)
    @inbounds for i ∈ 1:pf_pivot_count
        iRow = f.pf_pivot_index[i]
        value0 = vector.array[iRow]
        value1 = value0
        for k ∈ f.pf_start[i]:(f.pf_start[i + 1] - 1)
            value1 -= vector.array[f.pf_index[k]] * f.pf_value[k]
        end
        if value0 != 0.0 || value1 != 0.0
            if value0 == 0.0
                rhs_count += 1
                vector.index[rhs_count] = iRow
            end
            vector.array[iRow] = abs(value1) < kHighsTiny ? kHighsZero : value1
        end
    end
    vector.count = rhs_count
    pf_entries = f.pf_start[pf_pivot_count + 1] - 1
    vector.synthetic_tick += pf_pivot_count * 20 + pf_entries * 5
    if pf_entries / (pf_pivot_count + 1) < 5
        vector.synthetic_tick += pf_entries * 5
    end
    return vector
end

"""`HFactor::btranFT` — Forrest-Tomlin update solve (empty without update)."""
function btranFT!(f::HFactor, vector::HVector)
    rhs_count = vector.count
    pf_pivot_count = length(f.pf_pivot_index)
    rhs_synthetic_tick = 0.0
    @inbounds for i ∈ pf_pivot_count:-1:1
        pivotRow = f.pf_pivot_index[i]
        pivot_multiplier = vector.array[pivotRow]
        if pivot_multiplier != 0.0
            for k ∈ f.pf_start[i]:(f.pf_start[i + 1] - 1)
                iRow = f.pf_index[k]
                value0 = vector.array[iRow]
                value1 = value0 - pivot_multiplier * f.pf_value[k]
                if value0 == 0.0
                    rhs_count += 1
                    vector.index[rhs_count] = iRow
                end
                vector.array[iRow] = abs(value1) < kHighsTiny ? kHighsZero : value1
            end
            rhs_synthetic_tick += f.pf_start[i + 1] - f.pf_start[i]
        end
    end
    vector.synthetic_tick += rhs_synthetic_tick * 15 + pf_pivot_count * 10
    vector.count = rhs_count
    return vector
end

"""`HFactor::ftranL` — L solve (forward, sparse or hyper-sparse)."""
function ftranL!(f::HFactor, rhs::HVector, expected_density::Float64)
    current_density = 1.0 * rhs.count * f.inv_num_row
    sparse_solve = rhs.count < 0 || current_density > kHyperCancel ||
                   expected_density > kHyperFtranL
    if sparse_solve
        rhs_count = 0
        @inbounds for i ∈ 1:f.num_row
            pivotRow = f.l_pivot_index[i]
            pivot_multiplier = rhs.array[pivotRow]
            if abs(pivot_multiplier) > kHighsTiny
                rhs_count += 1
                rhs.index[rhs_count] = pivotRow
                for k ∈ f.l_start[i]:(f.l_start[i + 1] - 1)
                    rhs.array[f.l_index[k]] -= pivot_multiplier * f.l_value[k]
                end
            else
                rhs.array[pivotRow] = 0.0
            end
        end
        rhs.count = rhs_count
    else
        solveHyper!(rhs, f.num_row, f.l_pivot_lookup, f.l_pivot_index,
            EMPTY_FLOAT64_VEC, false, f.l_start, f.l_start, 1, f.l_index, f.l_value)
    end
    return rhs
end

"""`HFactor::btranL` — L^T solve (backward, sparse or hyper-sparse)."""
function btranL!(f::HFactor, rhs::HVector, expected_density::Float64)
    current_density = 1.0 * rhs.count * f.inv_num_row
    sparse_solve = rhs.count < 0 || current_density > kHyperCancel ||
                   expected_density > kHyperBtranL
    if sparse_solve
        rhs_count = 0
        @inbounds for i ∈ f.num_row:-1:1
            pivotRow = f.l_pivot_index[i]
            pivot_multiplier = rhs.array[pivotRow]
            if abs(pivot_multiplier) > kHighsTiny
                rhs_count += 1
                rhs.index[rhs_count] = pivotRow
                rhs.array[pivotRow] = pivot_multiplier
                for k ∈ f.lr_start[i]:(f.lr_start[i + 1] - 1)
                    rhs.array[f.lr_index[k]] -= pivot_multiplier * f.lr_value[k]
                end
            else
                rhs.array[pivotRow] = 0.0
            end
        end
        rhs.count = rhs_count
    else
        solveHyper!(rhs, f.num_row, f.l_pivot_lookup, f.l_pivot_index,
            EMPTY_FLOAT64_VEC, false, f.lr_start, f.lr_start, 1, f.lr_index, f.lr_value)
    end
    return rhs
end

"""`HFactor::ftranU` — U solve (backward, sparse or hyper-sparse)."""
function ftranU!(f::HFactor, rhs::HVector, expected_density::Float64, strategy::Val{S}=Val(ACTIVE_PIVOT_STRATEGY[])) where S
    if f.update_method == kUpdateMethodFt
        ftranFT!(f, rhs)
        tight!(rhs)
        pack!(rhs)
    end
    current_density = 1.0 * rhs.count * f.inv_num_row
    sparse_solve = rhs.count < 0 || current_density > kHyperCancel ||
                   expected_density > kHyperFtranU
    if sparse_solve
        rhs_synthetic_tick = 0.0
        rhs_count = 0
        u_pivot_count = length(f.u_pivot_index)
        @inbounds for i_logic ∈ u_pivot_count:-1:1
            f.u_pivot_index[i_logic] == 0 && continue
            pivotRow = f.u_pivot_index[i_logic]
            pivot_multiplier = rhs.array[pivotRow]
            if abs(pivot_multiplier) > kHighsTiny
                pivot_val = f.u_pivot_value[i_logic]
                pivot_inv = f.u_pivot_inv_value[i_logic]
                pivot_multiplier = apply_pivot(strategy, pivot_multiplier, pivot_val, pivot_inv)
                rhs_count += 1
                rhs.index[rhs_count] = pivotRow
                rhs.array[pivotRow] = pivot_multiplier
                if i_logic > f.num_row
                    rhs_synthetic_tick += f.u_last_p[i_logic] - f.u_start[i_logic]
                end
                for k ∈ f.u_start[i_logic]:(f.u_last_p[i_logic] - 1)
                    rhs.array[f.u_index[k]] -= pivot_multiplier * f.u_value[k]
                end
            else
                rhs.array[pivotRow] = 0.0
            end
        end
        rhs.count = rhs_count
        rhs.synthetic_tick += rhs_synthetic_tick * 15 +
                              (u_pivot_count - f.num_row) * 10
    else
        solveHyper!(rhs, f.num_row, f.u_pivot_lookup, f.u_pivot_index,
            f.u_pivot_value, true, f.u_start, f.u_last_p, 0, f.u_index,
            f.u_value, f.u_pivot_inv_value, strategy)
    end
    return rhs
end

"""`HFactor::btranU` — U^T solve (forward, sparse or hyper-sparse)."""
function btranU!(f::HFactor, rhs::HVector, expected_density::Float64, strategy::Val{S}=Val(ACTIVE_PIVOT_STRATEGY[])) where S
    current_density = 1.0 * rhs.count * f.inv_num_row
    sparse_solve = rhs.count < 0 || current_density > kHyperCancel ||
                   expected_density > kHyperBtranU
    if sparse_solve
        rhs_synthetic_tick = 0.0
        rhs_count = 0
        u_pivot_count = length(f.u_pivot_index)
        @inbounds for i_logic ∈ 1:u_pivot_count
            f.u_pivot_index[i_logic] == 0 && continue
            pivotRow = f.u_pivot_index[i_logic]
            pivot_multiplier = rhs.array[pivotRow]
            if abs(pivot_multiplier) > kHighsTiny
                pivot_val = f.u_pivot_value[i_logic]
                pivot_inv = f.u_pivot_inv_value[i_logic]
                pivot_multiplier = apply_pivot(strategy, pivot_multiplier, pivot_val, pivot_inv)
                rhs_count += 1
                rhs.index[rhs_count] = pivotRow
                rhs.array[pivotRow] = pivot_multiplier
                if i_logic > f.num_row
                    rhs_synthetic_tick += f.ur_lastp[i_logic] - f.ur_start[i_logic]
                end
                for k ∈ f.ur_start[i_logic]:(f.ur_lastp[i_logic] - 1)
                    rhs.array[f.ur_index[k]] -= pivot_multiplier * f.ur_value[k]
                end
            else
                rhs.array[pivotRow] = 0.0
            end
        end
        rhs.count = rhs_count
        rhs.synthetic_tick += rhs_synthetic_tick * 15 +
                              (u_pivot_count - f.num_row) * 10
    else
        solveHyper!(rhs, f.num_row, f.u_pivot_lookup, f.u_pivot_index,
            f.u_pivot_value, true, f.ur_start, f.ur_lastp, 0, f.ur_index,
            f.ur_value, f.u_pivot_inv_value, strategy)
    end
    if f.update_method == kUpdateMethodFt
        tight!(rhs)
        pack!(rhs)
        btranFT!(f, rhs)
        tight!(rhs)
    end
    return rhs
end

"""`HFactor::ftranCall` — `B x = b`."""
function ftranCall!(f::HFactor, vector::HVector, expected_density::Float64, strategy::Val{S}=Val(ACTIVE_PIVOT_STRATEGY[])) where S
    use_indices = vector.count >= 0
    ftranL!(f, vector, expected_density)
    ftranU!(f, vector, expected_density, strategy)
    use_indices && reIndex!(vector)
    return vector
end

"""`HFactor::btranCall` — `B^T x = b`."""
function btranCall!(f::HFactor, vector::HVector, expected_density::Float64, strategy::Val{S}=Val(ACTIVE_PIVOT_STRATEGY[])) where S
    use_indices = vector.count >= 0
    btranU!(f, vector, expected_density, strategy)
    btranL!(f, vector, expected_density)
    use_indices && reIndex!(vector)
    return vector
end
