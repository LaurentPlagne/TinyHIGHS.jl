// Référence C++ du même banc que `bench_julia.jl` : build, rebuild, FTRAN/BTRAN
// et chaîne d'updates, sur le même fichier de cas (format oracle HFactor).
//
//   hfactor_bench <cas.txt> [répétitions]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "util/HFactor.h"
#include "util/HighsSparseMatrix.h"
#include "util/HVector.h"

static long read_long() {
  long v;
  if (std::scanf("%ld", &v) != 1) {
    std::fprintf(stderr, "hfactor_bench : lecture entier échouée\n");
    std::exit(2);
  }
  return v;
}

static double read_double() {
  double v;
  if (std::scanf("%la", &v) != 1) {
    std::fprintf(stderr, "hfactor_bench : lecture flottant échouée\n");
    std::exit(2);
  }
  return v;
}

int main(int argc, char** argv) {
  const long repeats = argc > 1 ? std::atol(argv[1]) : 5;
  HighsSparseMatrix a;
  a.format_ = MatrixFormat::kColwise;
  a.num_row_ = static_cast<HighsInt>(read_long());
  a.num_col_ = static_cast<HighsInt>(read_long());
  const long num_basic = read_long();
  const long nnz = read_long();
  a.start_.assign(a.num_col_ + 1, 0);
  for (HighsInt i = 0; i <= a.num_col_; i++)
    a.start_[i] = static_cast<HighsInt>(read_long());
  a.index_.resize(nnz);
  for (long i = 0; i < nnz; i++)
    a.index_[i] = static_cast<HighsInt>(read_long());
  a.value_.resize(nnz);
  for (long i = 0; i < nnz; i++) a.value_[i] = read_double();
  std::vector<HighsInt> basic(num_basic);
  for (long i = 0; i < num_basic; i++)
    basic[i] = static_cast<HighsInt>(read_long());

  const HighsInt n = a.num_row_;
  auto now = []() { return std::chrono::steady_clock::now(); };
  auto ms_since = [&](std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(now() - t0).count();
  };
  auto median = [](std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };

  // Chauffe.
  {
    HFactor f;
    f.setup(a, basic);
    f.build();
    HVector v;
    v.setup(n);
    v.count = -1;
    f.ftranCall(v, 1.0);
    f.btranCall(v, 1.0);
  }

  // Les résultats sont consommés par `sink` pour interdire toute élimination.
  double sink = 0.0;

  std::vector<double> build_ms;
  for (long r = 0; r < repeats; r++) {
    HFactor f;
    auto t0 = now();
    f.setup(a, basic);
    f.build();
    build_ms.push_back(ms_since(t0));
    sink += f.getInvert().u_pivot_index.back();
  }

  HFactor f;
  f.setup(a, basic);
  f.build();
  const long per_repeat = 200;
  HVector v;
  v.setup(n);
  std::vector<double> solve_us;
  for (long r = 0; r < 3; r++) {
    auto t0 = now();
    for (long s = 1; s <= per_repeat; s++) {
      for (HighsInt i = 0; i < n; i++) v.array[i] = ((i * 7 + s) % 13 - 6) * 0.5;
      v.count = -1;
      f.ftranCall(v, 1.0);
      for (HighsInt i = 0; i < n; i++)
        v.array[i] = ((i * 5 + s) % 11 - 5) * 0.25;
      v.count = -1;
      f.btranCall(v, 1.0);
    }
    solve_us.push_back(ms_since(t0) * 1000.0 / (2 * per_repeat));
    sink += v.array[0] + v.array[n - 1];
  }

  const long n_updates = 100;
  std::vector<double> upd_us;
  for (long r = 0; r < 3; r++) {
    HFactor g;
    g.setup(a, basic);
    g.build();
    std::vector<HVector> aqs(n_updates);
    std::vector<HVector> eps(n_updates);
    std::vector<HighsInt> rows(n_updates);
    for (long t = 1; t <= n_updates; t++) {
      const HighsInt iRow = static_cast<HighsInt>((t - 1) % n);
      rows[t - 1] = iRow;
      HVector& aq = aqs[t - 1];
      aq.setup(n);
      aq.count = 1;
      aq.index[0] = iRow;
      aq.array[iRow] = 1.5;
      for (long k = 1; k <= 2; k++) {
        const HighsInt rr = static_cast<HighsInt>(((iRow + 1) * (k + 1) * 3) % n);
        bool dup = rr == iRow;
        for (HighsInt i = 0; i < aq.count; i++)
          if (aq.index[i] == rr) dup = true;
        if (dup) continue;
        aq.index[aq.count] = rr;
        aq.array[rr] = k == 1 ? 0.25 : -0.1;
        aq.count++;
      }
      aq.tight();
      aq.packFlag = true;
      aq.pack();
      HVector& ep = eps[t - 1];
      ep.setup(n);
      ep.count = 1;
      ep.index[0] = iRow;
      for (long k = 1; k <= 2; k++) {
        const HighsInt rr = static_cast<HighsInt>(((iRow + 1) * (k + 2) * 5) % n);
        bool dup = rr == iRow;
        for (HighsInt i = 0; i < ep.count; i++)
          if (ep.index[i] == rr) dup = true;
        if (dup) continue;
        ep.index[ep.count] = rr;
        ep.array[rr] = k == 1 ? 0.5 : -0.75;
        ep.count++;
      }
      ep.tight();
      ep.packFlag = true;
      ep.pack();
    }
    auto t0 = now();
    for (long t = 0; t < n_updates; t++) {
      HighsInt hint = 0;
      g.update(&aqs[t], &eps[t], &rows[t], &hint);
    }
    upd_us.push_back(ms_since(t0) * 1000.0 / n_updates);
    sink += g.getInvert().u_pivot_index.back();
  }

  std::vector<double> reb_ms;
  for (long r = 0; r < 3; r++) {
    HFactor g;
    g.setup(a, basic);
    g.build();
    g.refactor_info_.use = true;
    auto t0 = now();
    g.build();
    reb_ms.push_back(ms_since(t0));
    sink += g.getInvert().l_start.back();
  }

  std::printf("build      : min %8.3f ms   median %8.3f ms\n", build_ms.front(),
              median(build_ms));
  std::printf("rebuild    : min %8.3f ms   median %8.3f ms\n", reb_ms.front(),
              median(reb_ms));
  std::printf("ftran+btran: min %8.3f us/appel  median %8.3f us/appel\n",
              solve_us.front(), median(solve_us));
  std::printf("update     : min %8.3f us/update median %8.3f us/update\n",
              upd_us.front(), median(upd_us));
  std::printf("sink       : %g\n", sink);
  return 0;
}
