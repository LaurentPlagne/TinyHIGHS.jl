# Numerical constants of HiGHS, matching values in `lp_data/HConst.h` and
# `util/HFactorConst.h`.
const kHighsTiny = 1e-14
const kHighsZero = 1e-50

const kDefaultPivotThreshold = 0.1
const kMinPivotThreshold = 8e-4
const kMaxPivotThreshold = 0.5
const kDefaultPivotTolerance = 1e-10
const kMinPivotTolerance = 0.0
const kMaxPivotTolerance = 1.0

const kUpdateMethodFt = 1

const kHyperFtranL = 0.15
const kHyperFtranU = 0.10
const kHyperBtranL = 0.10
const kHyperBtranU = 0.15
const kHyperCancel = 0.05

const kRunningAverageMultiplier = 0.05
const kHyperPriceDensity = 0.1

# Pivot types (`lp_data/HConst.h:403`), encoded as Int8.
const kPivotIllegal = Int8(-1)
const kPivotLogical = Int8(0)
const kPivotUnit = Int8(1)
const kPivotRowSingleton = Int8(2)
const kPivotColSingleton = Int8(3)
const kPivotMarkowitz = Int8(4)

# Pivot inversion strategies in HFactor
@enum PivotStrategy begin
    kPivotBranching = 1
    kPivotBranchless = 2
    kPivotFdiv = 3
end

# `simplex/SimplexConst.h`: simplex and pricing strategies (`SimplexStrategy`, `SimplexPriceStrategy`).
const kSimplexStrategyDualPlain = 1
const kSimplexEdgeWeightStrategyChoose = -1
const kSimplexEdgeWeightStrategyDantzig = 0
const kSimplexEdgeWeightStrategyDevex = 1
const kSimplexEdgeWeightStrategySteepestEdge = 2
const kSimplexPriceStrategyCol = 0
const kSimplexPriceStrategyRowSwitchColSwitch = 3
# `lp_data/HConst.h`: LP scaling strategies and factor bounds.
const kSimplexScaleStrategyOff = 0
const kSimplexScaleStrategyChoose = 1
const kSimplexScaleStrategyEquilibration = 2
const kSimplexScaleStrategyForcedEquilibration = 3
const kSimplexScaleStrategyMaxValue = 4
const kDefaultAllowedMatrixPow2Scale = 20
const kNoRowChosen = -1
const kNoRowSought = -2
const kNoRayIndex = -1
const kNoRaySign = 0

# `simplex/SimplexConst.h`: weight modes and DSE acceptance threshold.
const kEdgeWeightDantzig = 0
const kEdgeWeightDevex = 1
const kEdgeWeightSteepestEdge = 2
const kAcceptDseWeightThreshold = 0.25
const kMinDualSteepestEdgeWeight = 1e-4
# Primal Devex: number of anomalous weights allowed before re-initialization.
const kAllowedNumBadDevexWeight = 3
const kBadDevexWeightFactor = 3.0

# `simplex/SimplexConst.h`: consequences of LP modification for `HEkk`
# (`HEkk::updateStatus`). `kNewRows` is also used for coefficient changes
# (`Highs::changeCoefficientInterface`) and invalidates full state.
const kLpActionScale = 0
const kLpActionNewCosts = 1
const kLpActionNewBounds = 2
const kLpActionNewBasis = 3
const kLpActionNewCols = 4
const kLpActionNewRows = 5

# `simplex/SimplexConst.h`: reasons for bad basis change.
const kBadBasisChangeAll = 0
const kBadBasisChangeSingular = 1
const kBadBasisChangeCycling = 2
const kBadBasisChangeFailedInfeasibilityProof = 3

# `simplex/SimplexConst.h`: reinversion reasons (`kRebuildReason*`).
const kRebuildReasonCleanup = -1
const kRebuildReasonNo = 0
const kRebuildReasonUpdateLimitReached = 1
const kRebuildReasonSyntheticClockSaysInvert = 2
const kRebuildReasonPossiblyOptimal = 3
const kRebuildReasonPossiblyPhase1Feasible = 4
const kRebuildReasonPossiblyPrimalUnbounded = 5
const kRebuildReasonPossiblyDualUnbounded = 6
const kRebuildReasonPossiblySingularBasis = 7
const kRebuildReasonPrimalInfeasibleInPrimalSimplex = 8
const kRebuildReasonChooseColumnFail = 9
const kRebuildReasonForceRefactor = 10
const kRebuildReasonExcessivePrimalValue = 11

# `util/HFactorConst.h`: simplex numerical robustness thresholds.
const kPivotThresholdChangeFactor = 5.0
const kNumericalTroubleTolerance = 1e-7
const kSyntheticTickReinversionMinUpdateCount = 50

# `HEkkDualRow`: working bounds for BFRT sorting.
const kInitialTotalChange = 1e-12
const kInitialRemainTheta = 1e100
const kMaxSelectTheta = 1e18

const kMCExtraEntriesMultiplier = 2
const kMRExtraEntriesMultiplier = 2
const kLFactorExtraEntriesMultiplier = 3
const kUFactorExtraVectors = 1000
const kUFactorExtraEntriesMultiplier = 3

# `lp_data/HConst.h`: HiGHS infinity representation, used by bounds and
# infeasibility measures (not a finite `1e30`).
const kHighsInf = Inf
const kHighsIInf = typemax(Int)
const kHighsIllegalInfeasibilityCount = -1
const kHighsIllegalInfeasibilityMeasure = kHighsInf

# `lp_data/HConst.h`: threshold beyond which a primal value is
# considered excessive (`HEkkDualRHS::updatePrimal`).
const kExcessivePrimalValue = 1e25

# `lp_data/HConst.h`: model status (`HighsModelStatus`, upstream order).
@enum ModelStatus kNotset = 0 kLoadError kModelError kPresolveError kSolveError kPostsolveError kModelEmpty kOptimal kInfeasible kUnboundedOrInfeasible kUnbounded kObjectiveBound kObjectiveTarget kTimeLimit kIterationLimit kUnknown kSolutionLimit kInterrupt kMemoryLimit

# `lp_data/HConst.h`: objective sense (`ObjSense`).
@enum ObjSense kMinimize = 1 kMaximize = -1

# `simplex/SimplexConst.h`: nonbasic flags and moves. Encoded values
# (-1/0/1, -99): do NOT offset them like indices.
const kNonbasicFlagTrue = Int8(1)
const kNonbasicFlagFalse = Int8(0)
const kIllegalFlagValue = Int8(-99)
const kNonbasicMoveUp = Int8(1)
const kNonbasicMoveDn = Int8(-1)
const kNonbasicMoveZe = Int8(0)
const kIllegalMoveValue = Int8(-99)

# `simplex/SimplexConst.h`: algorithm and phases. `kSolvePhaseMin` = -3;
# phase 2 is 2 (`HEkk::initialiseBound` tests this value).
@enum SimplexAlgorithm kNone = 0 kPrimal = 1 kDual = 2
const kSolvePhaseMin = -3
const kSolvePhaseError = -3
const kSolvePhaseExit = -2
const kSolvePhaseUnknown = -1
const kSolvePhaseOptimal = 0
const kSolvePhase1 = 1
const kSolvePhase2 = 2
const kSolvePhasePrimalInfeasibleCleanup = 3
const kSolvePhaseOptimalCleanup = 4
const kSolvePhaseTabooBasis = 5
