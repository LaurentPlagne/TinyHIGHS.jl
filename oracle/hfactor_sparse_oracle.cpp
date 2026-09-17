// Oracle des solves FTRAN/BTRAN à liste d'indices FOURNIE (`count >= 0`) :
// vérifie l'ordre de `index` après le solve, que les cas denses (`count = -1`)
// ne peuvent pas exercer. La source HFactor est celle des autres oracles ;
// protocole 0-based (le port Julia est 1-based).
//
// Entrée (stdin) :
//   num_row num_col nnz
//   a_start (num_col+1, 0-based) ; a_index ; a_value
//   basic_index (0-based, num_row)
//   num_updates ; (iRow packed_aq packed_ep)*
//   num_solve ; [count density ; (index value)*count ; value*count]*
// Sortie : 2 lignes par solve (FTRAN puis BTRAN) :
//   count index* count  array* (%a)
#include <cstdio>
#include <vector>

#include "util/HFactor.h"

static bool read_packed(HVector& v) {
  long long size, count, pack_count;
  if (fscanf(stdin, "%lld %lld", &size, &count) != 2) return false;
  v.setup((HighsInt)size);
  v.clear();
  v.count = (HighsInt)count;
  for (long long i = 0; i < count; i++) {
    long long x;
    if (fscanf(stdin, "%lld", &x) != 1) return false;
    v.index[i] = (HighsInt)x;
  }
  for (long long i = 0; i < size; i++) {
    double x;
    if (fscanf(stdin, "%lf", &x) != 1) return false;
    v.array[i] = x;
  }
  if (fscanf(stdin, "%lld", &pack_count) != 1) return false;
  v.packFlag = false;
  v.packCount = (HighsInt)pack_count;
  for (long long i = 0; i < pack_count; i++) {
    long long x;
    if (fscanf(stdin, "%lld", &x) != 1) return false;
    v.packIndex[i] = (HighsInt)x;
  }
  for (long long i = 0; i < pack_count; i++) {
    double x;
    if (fscanf(stdin, "%lf", &x) != 1) return false;
    v.packValue[i] = x;
  }
  return true;
}

int main() {
  long long num_row, num_col, nnz;
  while (fscanf(stdin, "%lld %lld %lld", &num_row, &num_col, &nnz) == 3) {
    HighsSparseMatrix a_matrix;
    a_matrix.format_ = MatrixFormat::kColwise;
    a_matrix.num_row_ = static_cast<HighsInt>(num_row);
    a_matrix.num_col_ = static_cast<HighsInt>(num_col);
    a_matrix.start_.resize(num_col + 1);
    for (long long i = 0; i <= num_col; i++) {
      long long x;
      if (fscanf(stdin, "%lld", &x) != 1) return 2;
      a_matrix.start_[i] = static_cast<HighsInt>(x);
    }
    a_matrix.index_.resize(nnz);
    for (long long i = 0; i < nnz; i++) {
      long long x;
      if (fscanf(stdin, "%lld", &x) != 1) return 2;
      a_matrix.index_[i] = static_cast<HighsInt>(x);
    }
    a_matrix.value_.resize(nnz);
    for (long long i = 0; i < nnz; i++)
      if (fscanf(stdin, "%lf", &a_matrix.value_[i]) != 1) return 2;
    std::vector<HighsInt> basic(num_row);
    for (long long i = 0; i < num_row; i++) {
      long long x;
      if (fscanf(stdin, "%lld", &x) != 1) return 2;
      basic[i] = static_cast<HighsInt>(x);
    }
    HFactor factor;
    factor.setup(a_matrix, basic);
    factor.build();
    long long num_updates;
    if (fscanf(stdin, "%lld", &num_updates) != 1) return 2;
    for (long long u = 0; u < num_updates; u++) {
      long long iRow;
      if (fscanf(stdin, "%lld", &iRow) != 1) return 2;
      HVector aq, ep;
      read_packed(aq);
      read_packed(ep);
      HighsInt row = (HighsInt)iRow;
      HighsInt hint = 0;
      factor.update(&aq, &ep, &row, &hint);
    }
    long long num_solve;
    if (fscanf(stdin, "%lld", &num_solve) != 1) return 2;
    for (long long k = 0; k < num_solve; k++) {
      long long count;
      double density;
      if (fscanf(stdin, "%lld %lf", &count, &density) != 2) return 2;
      std::vector<HighsInt> solve_index(count);
      std::vector<double> solve_value(count);
      for (long long i = 0; i < count; i++) {
        long long ix;
        double v;
        if (fscanf(stdin, "%lld %lf", &ix, &v) != 2) return 2;
        solve_index[i] = static_cast<HighsInt>(ix);
        solve_value[i] = v;
      }
      for (long long i = 0; i < count; i++)
        if (fscanf(stdin, "%lf", &solve_value[i]) != 1) return 2;
      HVector rhs;
      rhs.setup(a_matrix.num_row_);
      rhs.count = static_cast<HighsInt>(count);
      for (long long i = 0; i < count; i++) {
        rhs.index[i] = solve_index[i];
        rhs.array[solve_index[i]] = solve_value[i];
      }
      factor.ftranCall(rhs, density);
      printf("%lld", (long long)rhs.count);
      for (HighsInt i = 0; i < rhs.count; i++)
        printf(" %lld", (long long)rhs.index[i]);
      for (HighsInt i = 0; i < a_matrix.num_row_; i++)
        printf(" %a", rhs.array[i]);
      printf("\n");
      rhs.count = -1;          // force le remplissage complet de `clear`
      rhs.clear();
      rhs.count = static_cast<HighsInt>(count);
      for (long long i = 0; i < count; i++) {
        rhs.index[i] = solve_index[i];
        rhs.array[solve_index[i]] = solve_value[i];
      }
      factor.btranCall(rhs, density);
      printf("%lld", (long long)rhs.count);
      for (HighsInt i = 0; i < rhs.count; i++)
        printf(" %lld", (long long)rhs.index[i]);
      for (HighsInt i = 0; i < a_matrix.num_row_; i++)
        printf(" %a", rhs.array[i]);
      printf("\n");
    }
  }
  return 0;
}
