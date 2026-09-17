// Oracle `HighsRandom` : imprime une séquence fixe de tirages de la source
// INCHANGÉE (en-tête seul). Le port Julia doit la reproduire à l'identique.
//
//   julia_simplex/oracle/build.sh
//
// Sortie : une ligne par tirage, entiers en décimal, `fraction` en %a.
#include <cstdio>

#include "util/HighsRandom.h"

int main() {
  HighsRandom random(0);
  // `integer(sup)` pour plusieurs tailles, y compris les cas triviaux.
  const HighsInt sups[] = {1000, 1, 0, 2, 0x7fffffff, 65536, 3};
  for (HighsInt sup : sups) {
    for (int k = 0; k < 4; k++) printf("%lld\n", (long long)random.integer(sup));
  }
  // Réels en (0,1).
  for (int k = 0; k < 6; k++) printf("%a\n", random.fraction());
  // Shuffle d'une permutation de 8 éléments (indices 0-based).
  HighsInt data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  random.shuffle(data, 8);
  for (int k = 0; k < 8; k++) printf("%lld\n", (long long)data[k]);
  // Autre graine.
  random.initialise(42);
  for (int k = 0; k < 6; k++) printf("%lld\n", (long long)random.integer(100));
  for (int k = 0; k < 6; k++) printf("%a\n", random.fraction());
  return 0;
}
