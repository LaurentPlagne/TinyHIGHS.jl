// Oracle « échelles LP » : lit un LP au format de `dual_oracle.cpp` et imprime
// les facteurs calculés par `considerScaling`/`scaleLp` (source gelée) pour la
// stratégie demandée, afin de comparer le calcul du port à celui de HiGHS.
//
//   julia_simplex/oracle/build.sh
//
// Entrée (stdin) : le LP au format habituel, puis `end`.
// Sortie : `scaled <strategie_effective> <has_scaling>` puis
// `col <n> <facteurs %a>`, `row <n> <facteurs %a>` et
// `avalue <nnz> <valeurs %a>` (matrice échelonnée, pour comparer
// l'application des facteurs, pas seulement leur calcul).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "lp_data/HighsLp.h"
#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsOptions.h"

static bool read_double_vector(FILE* f, std::vector<double>& v, long n) {
  v.resize(n);
  for (long i = 0; i < n; i++)
    if (fscanf(f, "%lf", &v[i]) != 1) return false;
  return true;
}

static bool read_int_vector(FILE* f, std::vector<HighsInt>& v, long n) {
  v.resize(n);
  for (long i = 0; i < n; i++) {
    long long x;
    if (fscanf(f, "%lld", &x) != 1) return false;
    v[i] = (HighsInt)x;
  }
  return true;
}

int main(int argc, char** argv) {
  const HighsInt scale_strategy = argc > 1 ? (HighsInt)atoi(argv[1]) : 0;
  long long num_col, num_row, num_nz;
  if (fscanf(stdin, "%lld %lld %lld", &num_col, &num_row, &num_nz) != 3)
    return 2;
  std::vector<double> col_cost, col_lower, col_upper, row_lower, row_upper,
      a_value;
  std::vector<HighsInt> a_start, a_index;
  if (!read_double_vector(stdin, col_cost, num_col)) return 2;
  if (!read_double_vector(stdin, col_lower, num_col)) return 2;
  if (!read_double_vector(stdin, col_upper, num_col)) return 2;
  if (!read_double_vector(stdin, row_lower, num_row)) return 2;
  if (!read_double_vector(stdin, row_upper, num_row)) return 2;
  if (!read_int_vector(stdin, a_start, num_col + 1)) return 2;
  if (!read_int_vector(stdin, a_index, num_nz)) return 2;
  if (!read_double_vector(stdin, a_value, num_nz)) return 2;
  long long sense;
  double offset;
  if (fscanf(stdin, "%lld %lf", &sense, &offset) != 2) return 2;

  HighsOptions options;
  options.simplex_scale_strategy = scale_strategy;
  options.output_flag = false;
  HighsLp lp;
  lp.num_col_ = (HighsInt)num_col;
  lp.num_row_ = (HighsInt)num_row;
  lp.col_cost_ = col_cost;
  lp.col_lower_ = col_lower;
  lp.col_upper_ = col_upper;
  lp.row_lower_ = row_lower;
  lp.row_upper_ = row_upper;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.num_col_ = (HighsInt)num_col;
  lp.a_matrix_.num_row_ = (HighsInt)num_row;
  lp.a_matrix_.start_ = a_start;
  lp.a_matrix_.index_ = a_index;
  lp.a_matrix_.value_ = a_value;

  considerScaling(options, lp);
  printf("scaled %lld %d\n", (long long)lp.scale_.strategy,
         lp.scale_.has_scaling ? 1 : 0);
  printf("col %lld", (long long)lp.scale_.col.size());
  for (double v : lp.scale_.col) printf(" %a", v);
  printf("\nrow %lld", (long long)lp.scale_.row.size());
  for (double v : lp.scale_.row) printf(" %a", v);
  printf("\n");
  // Matrice échelonnée (valeurs colwise) : compare l'application des
  // facteurs, pas seulement leur calcul.
  printf("avalue %lld", (long long)lp.a_matrix_.value_.size());
  for (double v : lp.a_matrix_.value_) printf(" %a", v);
  printf("\n");
  return 0;
}
