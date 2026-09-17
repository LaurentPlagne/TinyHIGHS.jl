# Portage de `simplex/HEkkDualRHS.{h,cpp}` (licence MIT, HiGHS) — liste
# d'infaisabilités primales et mises à jour de `baseValue` pour le simplexe
# dual. Non porté : CHUZR multiple (`chooseMulti*`, PAMI) et le rapport
# `assessOptimality`.
#
# `work_infeasibility[iRow]` porte le carré de l'infaisabilité
# (`store_squared_primal_infeasibility = true`, valeur d'`initialiseControl`
# quand l'option `less_infeasible_DSE_check` est fausse, le défaut).

"""
    DualRHS(engine)

Miroir de `HEkkDualRHS` : liste des rangées primalement infaisables.
"""
mutable struct DualRHS
    engine::SimplexEngine
    workCutoff::Float64
    workCount::Int
    workMark::Vector{UInt8}
    workIndex::Vector{Int}
    work_infeasibility::Vector{Float64}
end

function DualRHS(e::SimplexEngine)
    num_row = e.lp.num_row
    return DualRHS(e, 0.0, 0, zeros(UInt8, num_row), zeros(Int, num_row),
        zeros(num_row))
end

"""Réinitialise un `DualRHS` pour une nouvelle résolution — miroir du constructeur."""
function reset!(rhs::DualRHS, e::SimplexEngine)
    num_row = e.lp.num_row
    rhs.engine = e
    rhs.workCutoff = 0.0
    rhs.workCount = 0
    if length(rhs.workMark) != num_row
        resize!(rhs.workMark, num_row)
        resize!(rhs.workIndex, num_row)
        resize!(rhs.work_infeasibility, num_row)
    end
    fill!(rhs.workMark, 0x00)
    fill!(rhs.workIndex, 0)
    fill!(rhs.work_infeasibility, 0.0)
    return rhs
end

"""
    choose_normal!(d)

`HEkkDualRHS::chooseNormal` : rangée de mérite maximal (infaisabilité / poids,
poids unitaires en Dantzig), en partant d'un indice tiré au hasard — l'ordre de
balayage est découpé en deux sections `[start, fin) ∪ [0, start)`. Rend
`kNoRowChosen` si la liste est vide.
"""
function choose_normal!(d::DualRHS)
    d.workCount == 0 && return kNoRowChosen
    e = d.engine
    edge_weight = e.dual_edge_weight
    if d.workCount < 0
        # Mode dense : `workCount = -numRow`.
        num_row = -d.workCount
        random_start = integer(e.random, num_row)
        best_merit = 0.0
        best_index = kNoRowChosen
        for section ∈ 0:1
            start = section == 0 ? random_start : 0
            stop = section == 0 ? num_row : random_start
            for iRow ∈ (start + 1):stop
                if d.work_infeasibility[iRow] > kHighsZero
                    my_infeas = d.work_infeasibility[iRow]
                    my_weight = edge_weight[iRow]
                    if best_merit * my_weight < my_infeas
                        best_merit = my_infeas / my_weight
                        best_index = iRow
                    end
                end
            end
        end
        return best_index
    end
    # Mode creux.
    random_start = integer(e.random, d.workCount)
    best_merit = 0.0
    best_index = kNoRowChosen
    for section ∈ 0:1
        start = section == 0 ? random_start : 0
        stop = section == 0 ? d.workCount : random_start
        for i ∈ (start + 1):stop
            iRow = d.workIndex[i]
            if d.work_infeasibility[iRow] > kHighsZero
                my_infeas = d.work_infeasibility[iRow]
                my_weight = edge_weight[iRow]
                if best_merit * my_weight < my_infeas
                    best_merit = my_infeas / my_weight
                    best_index = iRow
                end
            end
        end
    end
    create_list_again = best_index == kNoRowChosen ? d.workCutoff > 0 :
                        best_merit <= d.workCutoff * 0.99
    if create_list_again
        create_infeas_list!(d, 0.0)
        return choose_normal!(d)
    end
    return best_index
end

"""
    update_primal!(d, column, theta)

`HEkkDualRHS::updatePrimal` : `baseValue .-= theta * column` et remise à jour
des infaisabilités. Rend `false` si une valeur primale dépasse
`kExcessivePrimalValue` (l'appelant arme alors une ré-inversion).
"""
function update_primal!(d::DualRHS, column::HVector, theta::Float64)
    e = d.engine
    info = e.info
    num_row = e.lp.num_row
    tp = e.options.primal_feasibility_tolerance
    use_dense = column.count < 0 || column.count > 0.4 * num_row
    to_entry = use_dense ? num_row : column.count
    num_excessive = 0
    @inbounds for iEntry ∈ 1:to_entry
        iRow = use_dense ? iEntry : column.index[iEntry]
        info.baseValue[iRow] -= theta * column.array[iRow]
        lower = info.baseLower[iRow]
        upper = info.baseUpper[iRow]
        value = info.baseValue[iRow]
        inf1 = ifelse(value < lower - tp, lower - value, 0.0)
        inf2 = ifelse(value > upper + tp, value - upper, 0.0)
        primal_infeasibility = inf1 + inf2
        d.work_infeasibility[iRow] = info.store_squared_primal_infeasibility ?
                                     primal_infeasibility^2 :
                                     abs(primal_infeasibility)
        if value <= -kExcessivePrimalValue || value >= kExcessivePrimalValue
            num_excessive += 1
        end
    end
    return num_excessive == 0
end

"""`HEkkDualRHS::updatePivots` — valeur de la rangée pivot et infaisabilité."""
function update_pivots!(d::DualRHS, iRow::Int, value::Float64)
    info = d.engine.info
    tp = d.engine.options.primal_feasibility_tolerance
    info.baseValue[iRow] = value
    lower = info.baseLower[iRow]
    upper = info.baseUpper[iRow]
    inf1 = ifelse(value < lower - tp, lower - value, 0.0)
    inf2 = ifelse(value > upper + tp, value - upper, 0.0)
    primal_infeasibility = inf1 + inf2
    d.work_infeasibility[iRow] = info.store_squared_primal_infeasibility ?
                                 primal_infeasibility^2 :
                                 abs(primal_infeasibility)
    return d
end

"""`HEkkDualRHS::updateInfeasList` — ajoute les rangées devenues infaisables."""
function update_infeas_list!(d::DualRHS, column::HVector)
    d.workCount < 0 && return d          # dense : rien à tenir à jour
    e = d.engine
    edge_weight = e.dual_edge_weight
    if d.workCutoff <= 0
        @inbounds for i ∈ 1:column.count
            iRow = column.index[i]
            if d.workMark[iRow] == 0 && d.work_infeasibility[iRow] != 0
                d.workCount += 1
                d.workIndex[d.workCount] = iRow
                d.workMark[iRow] = 1
            end
        end
    else
        @inbounds for i ∈ 1:column.count
            iRow = column.index[i]
            if d.workMark[iRow] == 0 &&
               d.work_infeasibility[iRow] > edge_weight[iRow] * d.workCutoff
                d.workCount += 1
                d.workIndex[d.workCount] = iRow
                d.workMark[iRow] = 1
            end
        end
    end
    return d
end

"""`HEkkDualRHS::createArrayOfPrimalInfeasibilities`."""
function create_array_of_primal_infeasibilities!(d::DualRHS)
    e = d.engine
    info = e.info
    tp = e.options.primal_feasibility_tolerance
    store_sq = info.store_squared_primal_infeasibility
    @inbounds @simd for i ∈ 1:e.lp.num_row
        value = info.baseValue[i]
        lower = info.baseLower[i]
        upper = info.baseUpper[i]
        inf1 = ifelse(value < lower - tp, lower - value, 0.0)
        inf2 = ifelse(value > upper + tp, value - upper, 0.0)
        inf = inf1 + inf2
        d.work_infeasibility[i] = ifelse(store_sq, inf * inf, abs(inf))
    end
    return d
end

"""
    create_infeas_list!(d, columnDensity)

`HEkkDualRHS::createInfeasList` : reconstruit la liste, avec bascule
hyper-creuse quand elle est grande et la colonne peu dense (mérite
`infaisabilité / poids`, coupure par `nth_element`), puis mode dense au-delà de
20 % des rangées.
"""
function create_infeas_list!(d::DualRHS, columnDensity::Float64)
    e = d.engine
    num_row = e.lp.num_row
    edge_weight = e.dual_edge_weight
    dwork = e.scattered_dual_edge_weight
    fill!(d.workMark, 0x00)
    d.workCount = 0
    d.workCutoff = 0.0
    for iRow ∈ 1:num_row
        if d.work_infeasibility[iRow] != 0
            d.workMark[iRow] = 0x01
            d.workCount += 1
            d.workIndex[d.workCount] = iRow
        end
    end
    if d.workCount > max(num_row * 0.01, 500.0) && columnDensity < 0.05
        icutoff = trunc(Int, max(d.workCount * 0.001, 500.0))
        max_merit = 0.0
        i_put = 0
        for iRow ∈ 1:num_row
            if d.workMark[iRow] == 1
                my_merit = d.work_infeasibility[iRow] / edge_weight[iRow]
                max_merit = max(max_merit, my_merit)
                i_put += 1
                dwork[i_put] = -my_merit
            end
        end
        # `nth_element` : seule la valeur de rang `icutoff` (0-based) sert.
        cut_merit = -partialsort!(view(dwork, 1:i_put), icutoff + 1)
        d.workCutoff = min(max_merit * 0.99999, cut_merit * 1.00001)
        fill!(d.workMark, 0x00)
        d.workCount = 0
        for iRow ∈ 1:num_row
            if d.work_infeasibility[iRow] >= edge_weight[iRow] * d.workCutoff
                d.workCount += 1
                d.workIndex[d.workCount] = iRow
                d.workMark[iRow] = 0x01
            end
        end
        if d.workCount > icutoff * 1.5
            full_count = d.workCount
            d.workCount = icutoff
            for i ∈ (icutoff + 1):full_count
                iRow = d.workIndex[i]
                if d.work_infeasibility[iRow] > edge_weight[iRow] * cut_merit
                    d.workCount += 1
                    d.workIndex[d.workCount] = iRow
                else
                    d.workMark[iRow] = 0x00
                end
            end
        end
    end
    if d.workCount > 0.2 * num_row
        d.workCount = -num_row
        d.workCutoff = 0.0
    end
    return d
end
