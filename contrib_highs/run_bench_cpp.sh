#!/usr/bin/env bash
# ==============================================================================
# Banc de Rejeu C++ Autonome pour HiGHS (Warm-Start) — 100% Natif, ZÉRO Julia
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP_DIR="$SCRIPT_DIR/cpp"
CXX="${CXX:-clang++}"

# Détection de HiGHS C++ : build local patché en priorité, puis installation
# locale, puis artefact Julia officiel en dernier recours.  Un build CMake
# (HiGHS/build) garde ses headers dans l'arbre source et HConfig.h dans build;
# il ne peut donc pas être traité comme une installation packagée.
HIGHS_SOURCE="${HIGHS_SOURCE_DIR:-}"
HIGHS_BUILD="${HIGHS_BUILD_DIR:-}"

if [ -z "$HIGHS_BUILD" ] && [ -n "${HIGHS_INSTALL:-}" ]; then
    HIGHS_BUILD="$HIGHS_INSTALL"
fi

if [ -n "${HIGHS_DIR:-}" ]; then
    if [ -d "$HIGHS_DIR/build/lib" ]; then
        HIGHS_SOURCE="${HIGHS_SOURCE:-$HIGHS_DIR}"
        HIGHS_BUILD="$HIGHS_DIR/build"
    elif [ -d "$HIGHS_DIR/lib" ]; then
        HIGHS_BUILD="$HIGHS_DIR"
    fi
fi

LOCAL_HIGHS_DIR="$ROOT_DIR/../../HiGHS"
if [ -z "$HIGHS_BUILD" ] && [ -d "$LOCAL_HIGHS_DIR/build/lib" ]; then
    HIGHS_SOURCE="${HIGHS_SOURCE:-$(cd "$LOCAL_HIGHS_DIR" && pwd)}"
    HIGHS_BUILD="$(cd "$LOCAL_HIGHS_DIR/build" && pwd)"
fi
if [ -z "$HIGHS_BUILD" ] && [ -d "$LOCAL_HIGHS_DIR/install/lib" ]; then
    HIGHS_BUILD="$(cd "$LOCAL_HIGHS_DIR/install" && pwd)"
fi

# If the caller provided only HIGHS_BUILD_DIR, recover the source tree for a
# regular CMake build (public headers live one level above build/).
if [ -n "$HIGHS_BUILD" ] && [ -z "$HIGHS_SOURCE" ] &&
   [ -f "$HIGHS_BUILD/../highs/Highs.h" ] && [ -f "$HIGHS_BUILD/HConfig.h" ]; then
    HIGHS_SOURCE="$(cd "$HIGHS_BUILD/.." && pwd)"
fi

if [ -z "$HIGHS_BUILD" ]; then
    for artifact in "$HOME"/.julia/artifacts/*; do
        if [ -x "$artifact/bin/highs" ] && [ -d "$artifact/include/highs" ] && [ -d "$artifact/lib" ]; then
            HIGHS_BUILD="$artifact"
            break
        fi
    done
fi

PATCHED_INCLUDE_FLAGS=()
if [ -d "$HIGHS_BUILD/include/highs" ]; then
    # Packaged/install layout.
    PATCHED_INCLUDE_FLAGS=(-I "$HIGHS_BUILD/include/highs")
elif [ -n "$HIGHS_SOURCE" ] && [ -f "$HIGHS_SOURCE/highs/Highs.h" ] &&
     [ -f "$HIGHS_BUILD/HConfig.h" ]; then
    # CMake build layout: public headers in the source tree and generated
    # configuration header in the build tree.
    PATCHED_INCLUDE_FLAGS=(-I "$HIGHS_SOURCE/highs" -I "$HIGHS_BUILD")
fi

if [ -z "${ORIGINAL_HIGHS_LIB:-}" ]; then
    # Prefer the v1.15 artifact used by the benchmark. Julia may keep older
    # HiGHS artifacts alongside it; selecting the first filesystem match can
    # link an incompatible library (and, on macOS, a missing BLAS rpath).
    ORIGINAL_HIGHS_LIB="$(find "$HOME/.julia/artifacts" -maxdepth 4 \
        \( -name "libhighs.1.15.dylib" -o -name "libhighs.1.15.so" \) 2>/dev/null \
        | head -n 1 | xargs dirname 2>/dev/null || true)"
    if [ -z "$ORIGINAL_HIGHS_LIB" ]; then
        ORIGINAL_HIGHS_LIB="$(find "$HOME/.julia/artifacts" -maxdepth 4 \
            \( -name "libhighs*.dylib" -o -name "libhighs*.so" \) 2>/dev/null \
            | head -n 1 | xargs dirname 2>/dev/null || true)"
    fi
fi

if [ -z "$HIGHS_BUILD" ] || [ ! -d "$HIGHS_BUILD/lib" ] ||
   [ "${#PATCHED_INCLUDE_FLAGS[@]}" -eq 0 ]; then
    echo "Erreur: build HiGHS utilisable non trouvé."
    echo "Définissez HIGHS_BUILD_DIR=/chemin/vers/HiGHS/build"
    echo "ou HIGHS_DIR=/chemin/vers/HiGHS (arbre source avec build/)."
    exit 1
fi

PATCHED_LIB="$HIGHS_BUILD/lib/libhighs.dylib"
[ -f "$PATCHED_LIB" ] || PATCHED_LIB="$HIGHS_BUILD/lib/libhighs.so"

if [ ! -f "$PATCHED_LIB" ]; then
    echo "Erreur: aucune bibliothèque HiGHS trouvée dans $HIGHS_BUILD/lib."
    exit 1
fi

if [ -n "$HIGHS_SOURCE" ]; then
    HIGHS_TARGET_LABEL="HiGHS patché"
else
    HIGHS_TARGET_LABEL="HiGHS cible"
fi

echo "$HIGHS_TARGET_LABEL (headers) : ${HIGHS_SOURCE:-$HIGHS_BUILD}"
echo "$HIGHS_TARGET_LABEL (lib)     : $PATCHED_LIB"
if [ -n "$ORIGINAL_HIGHS_LIB" ]; then
    echo "HiGHS officiel (lib)   : $ORIGINAL_HIGHS_LIB"
else
    echo "HiGHS officiel (lib)   : indisponible"
fi

ORIGINAL_INCLUDE_FLAGS=()
if [ -n "$ORIGINAL_HIGHS_LIB" ] &&
   [ -f "$ORIGINAL_HIGHS_LIB/../include/highs/Highs.h" ]; then
    # The official artifact must be compiled with its own public headers:
    # mixing them with a newer source tree can produce an ABI mismatch even
    # though both libraries report the same HiGHS version.
    ORIGINAL_INCLUDE_FLAGS=(-I "$ORIGINAL_HIGHS_LIB/../include/highs")
else
    ORIGINAL_INCLUDE_FLAGS=("${PATCHED_INCLUDE_FLAGS[@]}")
fi

BIN_PATCHED="$CPP_DIR/replay_sequence"
BIN_ORIGINAL="$CPP_DIR/replay_sequence_original"

# 1. Compilation du binaire cible.  La sélection du build et son rpath
# peuvent changer sans modifier replay_sequence.cpp : on recompile donc ce
# petit driver à chaque appel afin d'éviter de comparer deux fois la même
# bibliothèque sous des noms différents.
echo "=== Compilation de replay_sequence ($HIGHS_TARGET_LABEL) ==="
"$CXX" -O3 -std=c++11 \
    "${PATCHED_INCLUDE_FLAGS[@]}" \
    "$CPP_DIR/replay_sequence.cpp" \
    -L "$HIGHS_BUILD/lib" \
    -lhighs -Wl,-rpath,"$HIGHS_BUILD/lib" \
    -o "$BIN_PATCHED"
echo "Compilation terminée : $BIN_PATCHED"
echo

# 2. Compilation du binaire original si la bibliothèque officielle existe
HAS_ORIGINAL=false
if [ -d "$ORIGINAL_HIGHS_LIB" ] && [ -f "$ORIGINAL_HIGHS_LIB/libhighs.dylib" -o -f "$ORIGINAL_HIGHS_LIB/libhighs.so" ]; then
    echo "=== Compilation de replay_sequence_original (HiGHS officiel v1.15.1) ==="
    ORIGINAL_BINARY_TMP="$BIN_ORIGINAL.tmp.$$"
    if "$CXX" -O3 -std=c++11 \
        "${ORIGINAL_INCLUDE_FLAGS[@]}" \
        "$CPP_DIR/replay_sequence.cpp" \
        -L "$ORIGINAL_HIGHS_LIB" \
        -lhighs -Wl,-rpath,"$ORIGINAL_HIGHS_LIB" \
        -o "$ORIGINAL_BINARY_TMP" 2>/dev/null; then
        mv "$ORIGINAL_BINARY_TMP" "$BIN_ORIGINAL"
    else
        rm -f "$ORIGINAL_BINARY_TMP"
        rm -f "$BIN_ORIGINAL"
        echo "Avertissement: compilation du binaire officiel impossible; comparaison désactivée."
    fi
    if [ -f "$BIN_ORIGINAL" ] && ("$BIN_ORIGINAL" 2>&1 || true) | grep -q "Usage"; then
        HAS_ORIGINAL=true
    fi
fi

echo "================================================================================"
echo " BANC DE REJEU C++ AUTONOMES DE SÉQUENCES LP (WARM-START)"
echo " Zéro dépendance Julia — C++11 natif via API officielle Highs"
echo "================================================================================"
echo

run_sequence() {
    local name="$1"
    local base_lp="$2"
    local ops="$3"
    local repeats="$4"

    echo "--------------------------------------------------------------------------------"
    echo ">>> Séquence : $name"
    echo "--------------------------------------------------------------------------------"

    if [ "$HAS_ORIGINAL" = true ]; then
        echo "[1/2] HiGHS Officiel v1.15.1 (sans patch) :"
        "$BIN_ORIGINAL" "$base_lp" "$ops" "$repeats"
        echo
        echo "[2/2] $HIGHS_TARGET_LABEL (SIMD reciprocal / branchless pivots) :"
        "$BIN_PATCHED" "$base_lp" "$ops" "$repeats"
    else
        "$BIN_PATCHED" "$base_lp" "$ops" "$repeats"
    fi
    echo
}

run_sequence "sequence_small (76 résolutions consécutives, base 88x107)" \
             "$ROOT_DIR/instances/sequences/sequence_small/base.lp" \
             "$ROOT_DIR/instances/sequences/sequence_small/operations.txt" 3

run_sequence "sequence_medium (100 résolutions consécutives, base 1240x1483)" \
             "$ROOT_DIR/instances/sequences/sequence_medium/base.lp" \
             "$ROOT_DIR/instances/sequences/sequence_medium/operations.txt" 2

echo "================================================================================"
echo " Fin du benchmark C++ natif."
echo "================================================================================"
