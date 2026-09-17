// Oracle HighsSparseMatrix : pilote la source HiGHS INCHANGÉE (produits, vues,
// pricing, computeDot/collectAj) et rend les résultats en %a pour comparaison
// bit-à-bit avec le port Julia (`julia_simplex/src/sparse_matrix.jl`).
//
// Protocole par cas :
//   <num_row> <num_col> <nnz>
//   <a_start (num_col+1, 0-based)> <a_index (nnz, 0-based)> <a_value (%a)>
//   <x (num_col, %a)>  <y (num_row, %a)>
//   <col_dense (num_row, %a)>                          (priceByColumn)
//   <col_sparse : size count index(count,0) array(size)>  (priceByRow)
//   <expected_density %a> <from_index 0-based> <switch_density %a>
//   <dot_array (num_row, %a)> <use_col 0-based>        (computeDot)
//   <collect state> <use_col 0-based> <multiplier %a>  (collectAj)
// Sortie par cas :
//   produit colwise, produitTranspose colwise, priceByColumn (count,index,array),
//   priceByColumn quad, computeDot, collectAj (count,index,array),
//   vue rowwise (start,index,value), produit rowwise, transpose rowwise,
//   priceByRow (count,index,array), priceByRowWithSwitch (count,index,array),
//   priceByRow quad, priceByRowWithSwitch quad,
//   partitionné : structure, priceByRowWithSwitch double et quad, mêmes sorties
//   après updates, puis échelles.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "util/HighsSparseMatrix.h"
#include "util/HVector.h"

[[noreturn]] static void fail(const char* what) {
  std::fprintf(stderr, "oracle matrix : %s\n", what);
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

static void read_sparse_state(HVector& v) {
  const long size = read_long();
  const long count = read_long();
  if (size < 0 || count > size) fail("état invalide");
  v.setup(static_cast<HighsInt>(size));
  v.count = static_cast<HighsInt>(count);
  for (long i = 0; i < count; i++)
    v.index[i] = static_cast<HighsInt>(read_long());
  for (long i = 0; i < size; i++) v.array[i] = read_double();
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

static void print_hvector(const HVector& v, HighsInt size) {
  std::printf("%d\n", static_cast<int>(v.count));
  for (HighsInt i = 0; i < v.count; i++)
    std::printf("%d ", static_cast<int>(v.index[i]));
  std::printf("\n");
  for (HighsInt i = 0; i < size; i++) std::printf("%a ", v.array[i]);
  std::printf("\n");
}

int main() {
  const long num_cases = read_long();
  for (long icase = 0; icase < num_cases; icase++) {
    HighsSparseMatrix a;
    a.format_ = MatrixFormat::kColwise;
    a.num_row_ = static_cast<HighsInt>(read_long());
    a.num_col_ = static_cast<HighsInt>(read_long());
    const long nnz = read_long();
    if (nnz < 0) fail("nnz négatif");
    a.start_.resize(a.num_col_ + 1);
    for (HighsInt i = 0; i <= a.num_col_; i++)
      a.start_[i] = static_cast<HighsInt>(read_long());
    a.index_.resize(nnz);
    for (long i = 0; i < nnz; i++)
      a.index_[i] = static_cast<HighsInt>(read_long());
    a.value_.resize(nnz);
    for (long i = 0; i < nnz; i++) a.value_[i] = read_double();
    // Copie colwise prise avant la conversion rowwise, pour la partition.
    HighsSparseMatrix a_col = a;

    std::vector<double> x(a.num_col_);
    for (HighsInt i = 0; i < a.num_col_; i++) x[i] = read_double();
    std::vector<double> y(a.num_row_);
    for (HighsInt i = 0; i < a.num_row_; i++) y[i] = read_double();

    std::vector<double> prod;
    a.product(prod, x);
    print_double_vector(prod);
    std::vector<double> prod_t;
    a.productTranspose(prod_t, y);
    print_double_vector(prod_t);

    HVector col_dense;
    read_sparse_state(col_dense);
    HVector result_pc;
    result_pc.setup(a.num_col_);
    a.priceByColumn(false, result_pc, col_dense);
    print_hvector(result_pc, a.num_col_);
    HVector result_pcq;
    result_pcq.setup(a.num_col_);
    a.priceByColumn(true, result_pcq, col_dense);
    print_hvector(result_pcq, a.num_col_);

    std::vector<double> dot_array(a.num_row_);
    for (HighsInt i = 0; i < a.num_row_; i++) dot_array[i] = read_double();
    const HighsInt use_col_dot = static_cast<HighsInt>(read_long());
    std::printf("%a\n", a.computeDot(dot_array, use_col_dot));

    HVector collect_state;
    read_sparse_state(collect_state);
    const HighsInt use_col_collect = static_cast<HighsInt>(read_long());
    const double multiplier = read_double();
    a.collectAj(collect_state, use_col_collect, multiplier);
    print_hvector(collect_state, a.num_row_);

    HVector col_sparse;
    read_sparse_state(col_sparse);
    const double expected_density = read_double();
    const HighsInt from_index = static_cast<HighsInt>(read_long());
    const double switch_density = read_double();

    a.ensureRowwise();
    print_int_vector(a.start_);
    print_int_vector(a.index_);
    print_double_vector(a.value_);
    std::vector<double> prod_r;
    a.product(prod_r, x);
    print_double_vector(prod_r);
    std::vector<double> prod_tr;
    a.productTranspose(prod_tr, y);
    print_double_vector(prod_tr);

    HVector result_pr;
    result_pr.setup(a.num_col_);
    a.priceByRow(false, result_pr, col_sparse);
    print_hvector(result_pr, a.num_col_);

    HVector result_pw;
    result_pw.setup(a.num_col_);
    a.priceByRowWithSwitch(false, result_pw, col_sparse, expected_density,
                           from_index, switch_density);
    print_hvector(result_pw, a.num_col_);

    HVector result_prq;
    result_prq.setup(a.num_col_);
    a.priceByRow(true, result_prq, col_sparse);
    print_hvector(result_prq, a.num_col_);

    HVector result_pwq;
    result_pwq.setup(a.num_col_);
    a.priceByRowWithSwitch(true, result_pwq, col_sparse, expected_density,
                           from_index, switch_density);
    print_hvector(result_pwq, a.num_col_);

    // Pricing partitionné : structure, prix, puis mises à jour var_in/var_out.
    std::vector<int8_t> in_partition(a.num_col_);
    for (HighsInt i = 0; i < a.num_col_; i++)
      in_partition[i] = static_cast<int8_t>(read_long());
    HighsSparseMatrix part;
    part.format_ = MatrixFormat::kColwise;
    part.createRowwisePartitioned(a_col, in_partition.data());
    print_int_vector(part.start_);
    print_int_vector(part.p_end_);
    print_int_vector(part.index_);
    print_double_vector(part.value_);
    HVector result_pp;
    result_pp.setup(a.num_col_);
    part.priceByRowWithSwitch(false, result_pp, col_sparse, expected_density,
                              from_index, switch_density);
    print_hvector(result_pp, a.num_col_);
    HVector result_ppq;
    result_ppq.setup(a.num_col_);
    part.priceByRowWithSwitch(true, result_ppq, col_sparse, expected_density,
                              from_index, switch_density);
    print_hvector(result_ppq, a.num_col_);

    const long n_updates = read_long();
    for (long u = 0; u < n_updates; u++) {
      HighsInt var_in = static_cast<HighsInt>(read_long());
      HighsInt var_out = static_cast<HighsInt>(read_long());
      part.update(var_in, var_out, a_col);
    }
    if (n_updates > 0) {
      print_int_vector(part.start_);
      print_int_vector(part.p_end_);
      print_int_vector(part.index_);
      print_double_vector(part.value_);
      HVector result_pu;
      result_pu.setup(a.num_col_);
      part.priceByRowWithSwitch(false, result_pu, col_sparse, expected_density,
                                from_index, switch_density);
      print_hvector(result_pu, a.num_col_);
      HVector result_puq;
      result_puq.setup(a.num_col_);
      part.priceByRowWithSwitch(true, result_puq, col_sparse, expected_density,
                                from_index, switch_density);
      print_hvector(result_puq, a.num_col_);
    }

    // Échelles (`HighsScale`) appliquées à la copie colwise.
    HighsScale scale;
    scale.col.resize(a.num_col_);
    for (HighsInt i = 0; i < a.num_col_; i++) scale.col[i] = read_double();
    scale.row.resize(a.num_row_);
    for (HighsInt i = 0; i < a.num_row_; i++) scale.row[i] = read_double();
    const HighsInt scale_col_idx = static_cast<HighsInt>(read_long());
    const double scale_col_val = read_double();
    const HighsInt scale_row_idx = static_cast<HighsInt>(read_long());
    const double scale_row_val = read_double();
    a_col.applyScale(scale);
    print_double_vector(a_col.value_);
    a_col.unapplyScale(scale);
    print_double_vector(a_col.value_);
    a_col.scaleCol(scale_col_idx, scale_col_val);
    print_double_vector(a_col.value_);
    a_col.scaleRow(scale_row_idx, scale_row_val);
    print_double_vector(a_col.value_);
  }
  return 0;
}
