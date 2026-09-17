// Oracle `HighsHashHelpers` : imprime une séquence fixe de hachages de la
// source INCHANGÉE (en-tête seul). Le port Julia doit la reproduire à
// l'identique (elle décide de la détection de cyclage du simplexe).
//
//   julia_simplex/oracle/build.sh
//
// Sortie : un hachage en %016llx par opération.
#include <cstdio>

#include "util/HighsHash.h"

int main() {
  uint64_t hash = 0;
  // Combinaisons sur plusieurs indices (bornes des blocs de 64).
  const HighsInt indices[] = {0, 1, 63, 64, 65, 127, 128, 200, 1000};
  for (HighsInt index : indices) {
    HighsHashHelpers::sparse_combine(hash, index);
    printf("%016llx\n", (unsigned long long)hash);
  }
  // Combinaisons inverses, dans un ordre différent.
  for (HighsInt index : {1000, 0, 128, 63, 65}) {
    HighsHashHelpers::sparse_inverse_combine(hash, index);
    printf("%016llx\n", (unsigned long long)hash);
  }
  // Séquence mêlée (comme une suite de pivots).
  for (HighsInt k = 0; k < 20; k++) {
    HighsHashHelpers::sparse_inverse_combine(hash, (k * 37) % 300);
    HighsHashHelpers::sparse_combine(hash, (k * 53 + 7) % 300);
    printf("%016llx\n", (unsigned long long)hash);
  }
  return 0;
}
