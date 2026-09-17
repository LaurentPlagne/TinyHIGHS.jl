#!/usr/bin/env bash
# Construit l'oracle C++ HVector : source HiGHS INCHANGÉE + harnais.
#
#   julia_simplex/oracle/build.sh
#
# Le commit des sources est consigné dans build/SOURCE_COMMIT et vérifié par
# les tests. La source attendue est le checkout local
# third_party/solvers/src/HiGHS (tag v1.15.1).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SRC="$ROOT/third_party/solvers/src/HiGHS"
INC="$ROOT/third_party/solvers/install/highs/include/highs"
# Référence NORMALE du solveur complet : libhighs SANS contraction FMA (le port
# n'utilise ni muladd ni fma ; un oracle contracté change les chemins). Voir
# docs/architecture/portage-julia-simplexe-highs.md §5.
ORACLE_LIB="$ROOT/third_party/solvers/install/highs-oracle/lib"
OUT="$HERE/build"
CXX="${CXX:-xcrun clang++}"

[ -f "$SRC/highs/util/HVectorBase.cpp" ] || {
    echo "source HiGHS absente : $SRC" >&2
    exit 1
}
[ -f "$INC/HConfig.h" ] || {
    echo "HConfig.h absent : construire HiGHS d'abord (scripts/solvers/build_custom_solvers.sh)" >&2
    exit 1
}

mkdir -p "$OUT"
# `-ffp-contract=off` : la source doit calculer `a + b*c` en deux opérations,
# comme le port Julia (qui n'utilise ni muladd ni fma, cf. plan §2.3). Sans ce
# drapeau, clang contracte en fmla et l'oracle diffère de 1 ULP.
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -DNDEBUG -ffp-contract=off -fno-fast-math \
    -I "$SRC/highs" -I "$INC" \
    "$SRC/highs/util/HVectorBase.cpp" "$HERE/hvector_oracle.cpp" \
    -o "$OUT/hvector_oracle"

# Oracle HFactor : HFactor*.cpp, HighsSparseMatrix et HVectorBase compilés
# localement avec les mêmes drapeaux ; le reste (logs, timers) vient de libhighs.
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -DNDEBUG -ffp-contract=off -fno-fast-math \
    -I "$SRC/highs" -I "$INC" \
    "$HERE/hfactor_oracle.cpp" \
    "$SRC/highs/util/HFactor.cpp" \
    "$SRC/highs/util/HFactorRefactor.cpp" \
    "$SRC/highs/util/HFactorUtils.cpp" \
    "$SRC/highs/util/HighsSparseMatrix.cpp" \
    "$SRC/highs/util/HVectorBase.cpp" \
    -L "$ROOT/third_party/solvers/install/highs/lib" -lhighs \
    -Wl,-rpath,"$ROOT/third_party/solvers/install/highs/lib" \
    -o "$OUT/hfactor_oracle"

# Banc C++ : mêmes sources locales que l'oracle HFactor, horloge interne.
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -DNDEBUG -ffp-contract=off -fno-fast-math \
    -I "$SRC/highs" -I "$INC" \
    "$HERE/hfactor_bench.cpp" \
    "$SRC/highs/util/HFactor.cpp" \
    "$SRC/highs/util/HFactorRefactor.cpp" \
    "$SRC/highs/util/HFactorUtils.cpp" \
    "$SRC/highs/util/HighsSparseMatrix.cpp" \
    "$SRC/highs/util/HVectorBase.cpp" \
    -L "$ROOT/third_party/solvers/install/highs/lib" -lhighs \
    -Wl,-rpath,"$ROOT/third_party/solvers/install/highs/lib" \
    -o "$OUT/hfactor_bench"

# Oracle matrice : mêmes sources locales que l'oracle HFactor.
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -DNDEBUG -ffp-contract=off -fno-fast-math \
    -I "$SRC/highs" -I "$INC" \
    "$HERE/matrix_oracle.cpp" \
    "$SRC/highs/util/HighsSparseMatrix.cpp" \
    "$SRC/highs/util/HVectorBase.cpp" \
    -L "$ROOT/third_party/solvers/install/highs/lib" -lhighs \
    -Wl,-rpath,"$ROOT/third_party/solvers/install/highs/lib" \
    -o "$OUT/matrix_oracle"


# Oracle des solves creux : `HFactor` local, liste d'indices fournie.
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -DNDEBUG -ffp-contract=off -fno-fast-math \
    -I "$SRC/highs" -I "$INC" \
    "$HERE/hfactor_sparse_oracle.cpp" \
    "$SRC/highs/util/HFactor.cpp" \
    "$SRC/highs/util/HFactorRefactor.cpp" \
    "$SRC/highs/util/HFactorUtils.cpp" \
    "$SRC/highs/util/HighsSparseMatrix.cpp" \
    "$SRC/highs/util/HVectorBase.cpp" \
    -L "$ROOT/third_party/solvers/install/highs/lib" -lhighs \
    -Wl,-rpath,"$ROOT/third_party/solvers/install/highs/lib" \
    -o "$OUT/hfactor_sparse_oracle"


# Oracle de hachage de base : `HighsHashHelpers` (en-tête seul).
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -DNDEBUG -ffp-contract=off -fno-fast-math \
    -I "$SRC/highs" -I "$INC" \
    "$HERE/hash_oracle.cpp" \
    -o "$OUT/hash_oracle"

# Oracle RNG : `HighsRandom` est en-tête seul (HighsHash.h).
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -DNDEBUG -ffp-contract=off -fno-fast-math \
    -I "$SRC/highs" -I "$INC" \
    "$HERE/random_oracle.cpp" \
    -o "$OUT/random_oracle"

# Oracle solveur dual : LP résolu par libhighs (source gelée) dans la
# configuration de la tranche M3b (simplexe dual Dantzig, presolve off).
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -DNDEBUG -ffp-contract=off -fno-fast-math \
    -I "$INC" \
    "$HERE/dual_oracle.cpp" \
    -L "$ORACLE_LIB" -lhighs \
    -Wl,-rpath,"$ORACLE_LIB" \
    -o "$OUT/dual_oracle"

# Oracle « séquence » : même LP que `dual_oracle`, muté entre plusieurs solves
# (bornes, coûts, coefficients, sens) — warm start du port (M5).
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -DNDEBUG -ffp-contract=off -fno-fast-math \
    -I "$SRC/highs" -I "$INC" \
    "$HERE/sequence_oracle.cpp" \
    -L "$ORACLE_LIB" -lhighs \
    -Wl,-rpath,"$ORACLE_LIB" \
    -o "$OUT/sequence_oracle"

# Oracle « échelles LP » : facteurs de `considerScaling`/`scaleLp` (M5b).
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -DNDEBUG -ffp-contract=off -fno-fast-math \
    -I "$SRC/highs" -I "$INC" \
    "$HERE/scaling_oracle.cpp" \
    -L "$ORACLE_LIB" -lhighs \
    -Wl,-rpath,"$ORACLE_LIB" \
    -o "$OUT/scaling_oracle"

git -C "$SRC" rev-parse HEAD > "$OUT/SOURCE_COMMIT"
git -C "$SRC" describe --tags >> "$OUT/SOURCE_COMMIT"

echo "oracles construits :"
ls -1 "$OUT/hvector_oracle" "$OUT/hfactor_oracle" "$OUT/hfactor_bench" \
    "$OUT/matrix_oracle" "$OUT/hfactor_sparse_oracle" "$OUT/random_oracle" \
    "$OUT/dual_oracle" "$OUT/sequence_oracle"
cat "$OUT/SOURCE_COMMIT"
