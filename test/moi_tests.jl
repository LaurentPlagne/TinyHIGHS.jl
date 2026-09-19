using MathOptInterface
const MOI = MathOptInterface

@testset "MathOptInterface — continuous LP adapter" begin
    model = TinyHiGHS.Optimizer()
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    row = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)], 0.0)
    c = MOI.add_constraint(model, row, MOI.GreaterThan(1.0))
    objective = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(2.0, x), MOI.ScalarAffineTerm(3.0, y)], 4.0)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 6.0 atol=1e-9
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 1.0 atol=1e-9
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 0.0 atol=1e-9
    @test MOI.get(model, MOI.ConstraintPrimal(), c) ≈ 1.0 atol=1e-9
    @test MOI.get(model, MOI.ConstraintDual(), c) ≈ 2.0 atol=1e-9

    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(3.0, x)], 0.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 0.0 atol=1e-9

    MOI.empty!(model)
    @test MOI.is_empty(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
end

@testset "MathOptInterface — bound-only LP and maximization" begin
    model = TinyHiGHS.Optimizer()
    x = MOI.add_variable(model)
    lower = MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
    upper = MOI.add_constraint(model, x, MOI.LessThan(4.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4.0 atol=1e-9
    @test MOI.get(model, MOI.ConstraintDual(), lower) ≈ 0.0 atol=1e-9
    @test MOI.get(model, MOI.ConstraintDual(), upper) ≈ -1.0 atol=1e-9
end

@testset "MathOptInterface — terminal statuses" begin
    infeasible = TinyHiGHS.Optimizer()
    x = MOI.add_variable(infeasible)
    MOI.add_constraint(infeasible, x, MOI.GreaterThan(2.0))
    MOI.add_constraint(infeasible, x, MOI.LessThan(1.0))
    MOI.set(infeasible, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    MOI.optimize!(infeasible)
    @test MOI.get(infeasible, MOI.TerminationStatus()) == MOI.INFEASIBLE
    @test MOI.get(infeasible, MOI.ResultCount()) == 0

    unbounded = TinyHiGHS.Optimizer()
    y = MOI.add_variable(unbounded)
    MOI.set(unbounded, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(unbounded, MOI.ObjectiveFunction{MOI.VariableIndex}(), y)
    MOI.optimize!(unbounded)
    @test MOI.get(unbounded, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
    @test MOI.get(unbounded, MOI.ResultCount()) == 0
end
