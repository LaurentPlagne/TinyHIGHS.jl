// Oracle HFactor : pilote `HFactor` de HiGHS (source INCHANGÉE) et rend l'état
// du facteur (L, U, pivots, carence de rang) et les résultats FTRAN/BTRAN, en
// hexadécimal C99 (%a) pour une comparaison bit-à-bit avec le port Julia.
//
// Compilé avec `-ffp-contract=off -fno-fast-math` (cf. oracle/build.sh) : la
// même arithmétique mul/add que le port, sans FMA implicite.
//
// Protocole (jeton par jeton) :
//   <nb_cas>
//   par cas :
//     <num_row> <num_col> <num_basic> <nnz>
//     <a_start (num_col+1, 0-based)>
//     <a_index (nnz, 0-based)>
//     <a_value (nnz, %a)>
//     <basic_index (num_basic, 0-based)>
//     <n_rhs>
//     <rhs (n_rhs × num_row, %a)>
//     <n_updates>
//     par update : <iRow 0-based> <aq packé> <ep packé>
//     <do_rebuild 0|1>
//   `aq`/`ep` packés : <size> <count> <index...> <array...> <packCount>
//                       <packIndex...> <packValue...>
//
// Sortie par cas :
//   <rank> <état> <solves>                    (build initial)
//   [<rank> <état> <solves>]                  (si n_updates > 0)
//   [<rank> <état> <solves>]                  (si do_rebuild = 1, sans updates)
// où <état> = l_start, l_index, l_value, l_pivot_index, u_pivot_index,
//             u_pivot_value, u_start, u_last_p, u_index, u_value, basic_index
// et <solves> = pour chaque rhs, une ligne FTRAN puis une ligne BTRAN (num_row %a).
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "util/HFactor.h"
#include "util/HighsSparseMatrix.h"
#include "util/HVector.h"

[[noreturn]] static void fail(const char* what) {
  std::fprintf(stderr, "oracle HFactor : %s\n", what);
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

static void print_int_vector(const std::vector<HighsInt>& v) {
  std::printf("%d\n", static_cast<int>(v.size()));
  for (HighsInt x : v) std::printf("%d ", static_cast<int>(x));
  std::printf("\n");
}

static void print_double_vector(const std::vector<double>& v) {
  std::printf("%d\n", static_cast<int>(v.size()));
  for (double x : v) std::printf("%a ", x);
  std::printf("\n");
}

// `HVector` est l'alias global de `HVectorBase<double>` (HVector.h).
static void read_packed_state(HVector& v) {
  const long size = read_long();
  const long count = read_long();
  if (size < 0 || count > size) fail("état packé invalide");
  v.setup(static_cast<HighsInt>(size));
  v.count = static_cast<HighsInt>(count);
  for (long i = 0; i < count; i++)
    v.index[i] = static_cast<HighsInt>(read_long());
  for (long i = 0; i < size; i++) v.array[i] = read_double();
  const long pack_count = read_long();
  v.packFlag = false;
  v.packCount = static_cast<HighsInt>(pack_count);
  for (long i = 0; i < pack_count; i++)
    v.packIndex[i] = static_cast<HighsInt>(read_long());
  for (long i = 0; i < pack_count; i++) v.packValue[i] = read_double();
}

static void dump_phase(const HFactor& factor, HighsInt rank_deficiency) {
  std::printf("%d\n", static_cast<int>(rank_deficiency));
  const InvertibleRepresentation invert = factor.getInvert();
  print_int_vector(invert.l_start);
  print_int_vector(invert.l_index);
  print_double_vector(invert.l_value);
  print_int_vector(invert.l_pivot_index);
  print_int_vector(invert.u_pivot_index);
  print_double_vector(invert.u_pivot_value);
  print_int_vector(invert.u_start);
  print_int_vector(invert.u_last_p);
  print_int_vector(invert.u_index);
  print_double_vector(invert.u_value);
  const HighsInt* base = factor.getBaseIndex();
  print_int_vector(std::vector<HighsInt>(base, base + factor.num_basic));
}

static void print_solves(const HFactor& factor, HighsInt num_row,
                         const std::vector<std::vector<double>>& rhs_list) {
  for (const auto& rhs : rhs_list) {
    HVector v;
    v.setup(num_row);
    v.count = -1;
    for (HighsInt i = 0; i < num_row; i++) v.array[i] = rhs[i];
    factor.ftranCall(v, 1.0);
    for (HighsInt i = 0; i < num_row; i++) std::printf("%a ", v.array[i]);
    std::printf("\n");

    HVector w;
    w.setup(num_row);
    w.count = -1;
    for (HighsInt i = 0; i < num_row; i++) w.array[i] = rhs[i];
    factor.btranCall(w, 1.0);
    for (HighsInt i = 0; i < num_row; i++) std::printf("%a ", w.array[i]);
    std::printf("\n");
  }
}

int main() {
  const long num_cases = read_long();
  for (long icase = 0; icase < num_cases; icase++) {
    HighsSparseMatrix a_matrix;
    a_matrix.format_ = MatrixFormat::kColwise;
    a_matrix.num_row_ = static_cast<HighsInt>(read_long());
    a_matrix.num_col_ = static_cast<HighsInt>(read_long());
    const long num_basic = read_long();
    const long nnz = read_long();
    if (nnz < 0 || num_basic < 0) fail("dimensions invalides");
    a_matrix.start_.resize(a_matrix.num_col_ + 1);
    for (HighsInt i = 0; i <= a_matrix.num_col_; i++)
      a_matrix.start_[i] = static_cast<HighsInt>(read_long());
    a_matrix.index_.resize(nnz);
    for (long i = 0; i < nnz; i++)
      a_matrix.index_[i] = static_cast<HighsInt>(read_long());
    a_matrix.value_.resize(nnz);
    for (long i = 0; i < nnz; i++) a_matrix.value_[i] = read_double();

    std::vector<HighsInt> basic_index(num_basic);
    for (long i = 0; i < num_basic; i++)
      basic_index[i] = static_cast<HighsInt>(read_long());

    const long n_rhs = read_long();
    std::vector<std::vector<double>> rhs_list(
        n_rhs, std::vector<double>(a_matrix.num_row_));
    for (long r = 0; r < n_rhs; r++)
      for (HighsInt i = 0; i < a_matrix.num_row_; i++)
        rhs_list[r][i] = read_double();

    HFactor factor;
    factor.setup(a_matrix, basic_index);
    const HighsInt rank_deficiency = factor.build();
    dump_phase(factor, rank_deficiency);
    print_solves(factor, a_matrix.num_row_, rhs_list);

    const long n_updates = read_long();
    for (long u = 0; u < n_updates; u++) {
      HighsInt iRow = static_cast<HighsInt>(read_long());
      HVector aq;
      HVector ep;
      read_packed_state(aq);
      read_packed_state(ep);
      HighsInt hint = 0;
      factor.update(&aq, &ep, &iRow, &hint);
    }
    if (n_updates > 0) {
      dump_phase(factor, 0);
      print_solves(factor, a_matrix.num_row_, rhs_list);
    }

    const long do_rebuild = read_long();
    if (do_rebuild && n_updates == 0) {
      factor.refactor_info_.use = true;
      const HighsInt rebuild_rank = factor.build();
      dump_phase(factor, rebuild_rank);
      print_solves(factor, a_matrix.num_row_, rhs_list);
    }
  }
  return 0;
}
