module TinyHiGHSMathOptInterfaceExt

import MathOptInterface as MOI
import TinyHiGHS
import TinyHiGHS: Optimizer, SimplexEngine, SimplexLp, SimplexOptions,
    SparseMatrix, ModelStatus, kHighsInf, kHighsIInf, kMinimize, kMaximize,
    kOptimal, kInfeasible, kUnbounded, kUnboundedOrInfeasible, kTimeLimit,
    kIterationLimit, solve!

include(joinpath(@__DIR__, "..", "src", "moi.jl"))

end
