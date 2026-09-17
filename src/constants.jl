# Constantes numériques de HiGHS, valeurs identiques à `lp_data/HConst.h` et
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

# Types de pivot (`lp_data/HConst.h:403`), encodés Int8.
const kPivotIllegal = Int8(-1)
const kPivotLogical = Int8(0)
const kPivotUnit = Int8(1)
const kPivotRowSingleton = Int8(2)
const kPivotColSingleton = Int8(3)
const kPivotMarkowitz = Int8(4)

# `simplex/SimplexConst.h` : stratégies (`SimplexStrategy`, `SimplexPriceStrategy`).
const kSimplexStrategyDualPlain = 1
const kSimplexEdgeWeightStrategyChoose = -1
const kSimplexEdgeWeightStrategyDantzig = 0
const kSimplexEdgeWeightStrategyDevex = 1
const kSimplexEdgeWeightStrategySteepestEdge = 2
const kSimplexPriceStrategyCol = 0
const kSimplexPriceStrategyRowSwitchColSwitch = 3
# `lp_data/HConst.h` : stratégies d'échelles LP et bornes de facteurs.
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

# `simplex/SimplexConst.h` : modes de poids et seuil d'acceptation DSE.
const kEdgeWeightDantzig = 0
const kEdgeWeightDevex = 1
const kEdgeWeightSteepestEdge = 2
const kAcceptDseWeightThreshold = 0.25
const kMinDualSteepestEdgeWeight = 1e-4
# Devex primal : nombre de poids aberrants tolérés avant ré-initialisation.
const kAllowedNumBadDevexWeight = 3
const kBadDevexWeightFactor = 3.0

# `simplex/SimplexConst.h` : conséquences d'une modification du LP pour `HEkk`
# (`HEkk::updateStatus`). `kNewRows` sert aussi aux changements de coefficient
# (`Highs::changeCoefficientInterface`) et invalide tout l'état.
const kLpActionScale = 0
const kLpActionNewCosts = 1
const kLpActionNewBounds = 2
const kLpActionNewBasis = 3
const kLpActionNewCols = 4
const kLpActionNewRows = 5

# `simplex/SimplexConst.h` : raisons de mauvais changement de base.
const kBadBasisChangeAll = 0
const kBadBasisChangeSingular = 1
const kBadBasisChangeCycling = 2
const kBadBasisChangeFailedInfeasibilityProof = 3

# `simplex/SimplexConst.h` : raisons de ré-inversion (`kRebuildReason*`).
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

# `util/HFactorConst.h` : seuils de robustesse du simplexe.
const kPivotThresholdChangeFactor = 5.0
const kNumericalTroubleTolerance = 1e-7
const kSyntheticTickReinversionMinUpdateCount = 50

# `HEkkDualRow` : bornes de travail du tri de BFRT.
const kInitialTotalChange = 1e-12
const kInitialRemainTheta = 1e100
const kMaxSelectTheta = 1e18

const kMCExtraEntriesMultiplier = 2
const kMRExtraEntriesMultiplier = 2
const kLFactorExtraEntriesMultiplier = 3
const kUFactorExtraVectors = 1000
const kUFactorExtraEntriesMultiplier = 3

# `lp_data/HConst.h` : infini de HiGHS, porté par les bornes et les mesures
# d'infaisabilité (ce n'est pas un `1e30` fini).
const kHighsInf = Inf
const kHighsIInf = typemax(Int)
const kHighsIllegalInfeasibilityCount = -1
const kHighsIllegalInfeasibilityMeasure = kHighsInf

# `lp_data/HConst.h` : valeur au-delà de laquelle une valeur primale est
# considérée comme excessive (`HEkkDualRHS::updatePrimal`).
const kExcessivePrimalValue = 1e25

# `lp_data/HConst.h` : statut du modèle (`HighsModelStatus`, ordre source).
@enum ModelStatus kNotset = 0 kLoadError kModelError kPresolveError kSolveError kPostsolveError kModelEmpty kOptimal kInfeasible kUnboundedOrInfeasible kUnbounded kObjectiveBound kObjectiveTarget kTimeLimit kIterationLimit kUnknown kSolutionLimit kInterrupt kMemoryLimit

# `lp_data/HConst.h` : sens de l'objectif (`ObjSense`).
@enum ObjSense kMinimize = 1 kMaximize = -1

# `simplex/SimplexConst.h` : drapeaux et mouvements non basiques. Valeurs
# **encodées** (-1/0/1, -99) : ne pas les décaler d'un cran comme les indices.
const kNonbasicFlagTrue = Int8(1)
const kNonbasicFlagFalse = Int8(0)
const kIllegalFlagValue = Int8(-99)
const kNonbasicMoveUp = Int8(1)
const kNonbasicMoveDn = Int8(-1)
const kNonbasicMoveZe = Int8(0)
const kIllegalMoveValue = Int8(-99)

# `simplex/SimplexConst.h` : algorithme et phases. `kSolvePhaseMin` = -3 ; la
# phase 2 vaut 2 (`HEkk::initialiseBound` teste cette valeur).
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
