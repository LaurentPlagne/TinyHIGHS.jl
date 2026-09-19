#!/usr/bin/env bash
# ==============================================================================
# Banc de Rejeu C++ Autonome pour HiGHS (Warm-Start) — 100% Natif, ZÉRO Julia
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP_DIR="$SCRIPT_DIR/cpp"
CXX="${CXX:-clang++}"

# Détection de l'installation HiGHS C++ locale (patchée ou custom)
if [ -z "${HIGHS_INSTALL:-}" ]; then
    HIGHS_INSTALL="${HIGHS_DIR:-}"
fi
if [ -z "$HIGHS_INSTALL" ] && [ -d "$SCRIPT_DIR/../../HiGHS/install" ]; then
    HIGHS_INSTALL="$(cd "$SCRIPT_DIR/../../HiGHS/install" && pwd)"
fi
if [ -z "$HIGHS_INSTALL" ]; then
    for artifact in "$HOME"/.julia/artifacts/*; do
        if [ -x "$artifact/bin/highs" ] && [ -d "$artifact/include/highs" ] && [ -d "$artifact/lib" ]; then
            HIGHS_INSTALL="$artifact"
            break
        fi
    done
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

if [ -z "$HIGHS_INSTALL" ] || [ ! -d "$HIGHS_INSTALL" ]; then
    echo "Erreur: Répertoire d'installation de HiGHS non trouvé."
    echo "Définissez HIGHS_INSTALL=/chemin/vers/install/highs ou HIGHS_DIR=/chemin/vers/build/highs"
    exit 1
fi

BIN_PATCHED="$CPP_DIR/replay_sequence"
BIN_ORIGINAL="$CPP_DIR/replay_sequence_original"

# 1. Compilation du binaire patché
if [ ! -f "$BIN_PATCHED" ] || [ "$CPP_DIR/replay_sequence.cpp" -nt "$BIN_PATCHED" ]; then
    echo "=== Compilation de replay_sequence (HiGHS patché) ==="
    $CXX -O3 -std=c++11 \
        -I "$HIGHS_INSTALL/include/highs" \
        -L "$HIGHS_INSTALL/lib" \
        -lhighs -Wl,-rpath,"$HIGHS_INSTALL/lib" \
        "$CPP_DIR/replay_sequence.cpp" \
        -o "$BIN_PATCHED"
    echo "Compilation terminée : $BIN_PATCHED"
    echo
fi

# 2. Compilation du binaire original si la bibliothèque officielle existe
HAS_ORIGINAL=false
if [ -d "$ORIGINAL_HIGHS_LIB" ] && [ -f "$ORIGINAL_HIGHS_LIB/libhighs.dylib" -o -f "$ORIGINAL_HIGHS_LIB/libhighs.so" ]; then
    if [ ! -f "$BIN_ORIGINAL" ] || [ "$CPP_DIR/replay_sequence.cpp" -nt "$BIN_ORIGINAL" ]; then
        echo "=== Compilation de replay_sequence_original (HiGHS officiel v1.15.1) ==="
        $CXX -O3 -std=c++11 \
        -I "$HIGHS_INSTALL/include/highs" \
        -L "$ORIGINAL_HIGHS_LIB" \
        -lhighs -Wl,-rpath,"$ORIGINAL_HIGHS_LIB" \
        "$CPP_DIR/replay_sequence.cpp" \
        -o "$BIN_ORIGINAL" 2>/dev/null || true
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
        echo "[2/2] HiGHS Patché (Short-circuit unit diagonal pivots) :"
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
