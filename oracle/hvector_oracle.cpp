// Oracle HVector : pilote la source HiGHS INCHANGÉE (`HVectorBase`) et rend
// l'état après chaque opération en hexadécimal C99 (%a) pour une comparaison
// bit-à-bit avec le port Julia. Voir
// docs/architecture/portage-julia-simplexe-highs.md §4.1.
//
// Protocole (jeton par jeton) :
//   <nb_cas>
//   pour chaque cas :
//     <size> <count>
//     <index_0based ...>            (count jetons, omis si count <= 0)
//     <valeur_hex ...>              (size jetons)
//     <op> [charge]
//       tight | clear | reindex | norm2 | pack
//       saxpy <pivotX_hex> <psize> <pcount> <pidx...> <pvaleur...>
//       copy  <dsize> <dcount> <didx...> <dvaleur...>
//   sortie par cas (le cas `copy` rend la destination) :
//     <count> <index_0based...>
//     <valeur_hex...>
//     <packFlag 0|1> <packCount> <packIndex_0based...> <packValue_hex...>
//     <synthetic_tick_hex> <norm2_hex>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "util/HVectorBase.h"

using HVectorD = HVectorBase<double>;

[[noreturn]] static void fail(const char* what) {
  std::fprintf(stderr, "oracle HVector : %s\n", what);
  std::exit(2);
}

static long read_long() {
  long v;
  if (std::scanf("%ld", &v) != 1) fail("lecture entier");
  return v;
}

static double read_double() {
  double v;
  if (std::scanf("%la", &v) != 1) fail("lecture flottant hex");
  return v;
}

static void read_state(HVectorD& v) {
  const long size = read_long();
  const long count = read_long();
  if (size < 0 || count > size) fail("état invalide");
  v.setup(static_cast<HighsInt>(size));
  v.count = static_cast<HighsInt>(count);
  for (long i = 0; i < count; i++)
    v.index[i] = static_cast<HighsInt>(read_long());
  for (long i = 0; i < size; i++) v.array[i] = read_double();
}

static void print_state(const HVectorD& v) {
  std::printf("%d\n", static_cast<int>(v.count));
  for (HighsInt i = 0; i < v.count; i++)
    std::printf("%d ", static_cast<int>(v.index[i]));
  std::printf("\n");
  for (HighsInt i = 0; i < v.size; i++) std::printf("%a ", v.array[i]);
  std::printf("\n");
  std::printf("%d %d\n", v.packFlag ? 1 : 0, static_cast<int>(v.packCount));
  for (HighsInt i = 0; i < v.packCount; i++)
    std::printf("%d ", static_cast<int>(v.packIndex[i]));
  std::printf("\n");
  for (HighsInt i = 0; i < v.packCount; i++)
    std::printf("%a ", v.packValue[i]);
  std::printf("\n");
  std::printf("%a\n", v.synthetic_tick);
  std::printf("%a\n", v.norm2());
}

int main() {
  const long num_cases = read_long();
  char op[32];
  for (long icase = 0; icase < num_cases; icase++) {
    HVectorD v;
    read_state(v);
    if (std::scanf("%31s", op) != 1) fail("lecture opération");
    if (std::strcmp(op, "tight") == 0) {
      v.tight();
      print_state(v);
    } else if (std::strcmp(op, "clear") == 0) {
      v.clear();
      print_state(v);
    } else if (std::strcmp(op, "reindex") == 0) {
      v.reIndex();
      print_state(v);
    } else if (std::strcmp(op, "pack") == 0) {
      v.packFlag = true;
      v.pack();
      print_state(v);
    } else if (std::strcmp(op, "norm2") == 0) {
      print_state(v);
    } else if (std::strcmp(op, "saxpy") == 0) {
      const double pivotX = read_double();
      HVectorD pivot;
      read_state(pivot);
      v.saxpy(pivotX, &pivot);
      print_state(v);
    } else if (std::strcmp(op, "copy") == 0) {
      HVectorD dest;
      read_state(dest);
      dest.copy(&v);
      print_state(dest);
    } else {
      fail("opération inconnue");
    }
  }
  return 0;
}
