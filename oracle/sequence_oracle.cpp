// Oracle « séquence » : rejoue un LP muté entre plusieurs résolutions, avec
// HiGHS (libhighs de la source gelée sans contraction FMA) dans la
// configuration du port : `presolve=off`, `simplex_scale_strategy` réglable
// (0 = pas d'échelles), simplexe dual/primal selon les arguments.
//
//   julia_simplex/oracle/build.sh
//
// Entrée (stdin) : le LP au format de `dual_oracle.cpp`, puis des opérations,
// une par ligne, jusqu'à `end` :
//   change_col_bounds c lo up
//   change_cols_bounds n c1 lo1 up1 c2 lo2 up2 ...
//   change_row_bounds r lo up
//   change_rows_bounds n r1 lo1 up1 ...
//   change_col_cost c value
//   change_cols_cost n c1 v1 c2 v2 ...
//   change_coeff r c value
//   change_sense s
//   limit n
//   solve
//   end
// Indices 1-based, flottants en %a (comme le LP). Chaque `solve` imprime
// `solve <n> <statut> <objectif> <itérations>` et, si `print_vectors` est
// demandé, les quatre vecteurs (`colvalue`, `coldual`, `rowvalue`, `rowdual`).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "interfaces/highs_c_api.h"

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

// La source gelée appelle `debugPrimalSteepestEdgeWeights` en stratégie
// steepest edge et imprime par `printf` hors journal : détourner stdout
// pendant les solves pour que les tests ne lisent que les résultats.
static int silence_stdout() {
  fflush(stdout);
  const int fd = dup(fileno(stdout));
  if (fd >= 0) {
    FILE* sink = freopen("/dev/null", "w", stdout);
    (void)sink;
  }
  return fd;
}

static void restore_stdout(int fd) {
  fflush(stdout);
  if (fd >= 0) {
    dup2(fd, fileno(stdout));
    close(fd);
  }
}

int main(int argc, char** argv) {
  // Configuration, mêmes arguments que `dual_oracle.cpp` :
  // scale, edge, cost_perturbation, iteration_limit, dse_weight_error_threshold,
  // primal_edge_weight_strategy, primal_bound_perturbation, simplex_strategy,
  // puis `print_vectors` (0/1).
  const HighsInt scale_strategy = argc > 1 ? (HighsInt)atoi(argv[1]) : 0;
  const HighsInt edge_weight_strategy = argc > 2 ? (HighsInt)atoi(argv[2]) : 0;
  const double cost_perturbation = argc > 3 ? atof(argv[3]) : 0.0;
  const HighsInt iteration_limit = argc > 4 ? (HighsInt)atoi(argv[4]) : 0;
  const double dse_weight_error_threshold = argc > 5 ? atof(argv[5]) : -1.0;
  const HighsInt primal_edge_weight_strategy =
      argc > 6 ? (HighsInt)atoi(argv[6]) : -1;
  const double primal_bound_perturbation = argc > 7 ? atof(argv[7]) : -1.0;
  const HighsInt simplex_strategy = argc > 8 ? (HighsInt)atoi(argv[8]) : -1;
  const int print_vectors = argc > 9 ? atoi(argv[9]) : 0;
  // Diagnostic (harnais) : `highs_analysis_level` > 0 active le rapport de
  // synthèse de HiGHS, imprimé par `printf` hors journal — le détournement de
  // stdout est alors désactivé pour le laisser passer.
  const HighsInt analysis_level = argc > 10 ? (HighsInt)atoi(argv[10]) : 0;

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

  void* highs = Highs_create();
  Highs_setBoolOptionValue(highs, "output_flag", 0);
  Highs_setStringOptionValue(highs, "solver", "simplex");
  Highs_setStringOptionValue(highs, "presolve", "off");
  Highs_setStringOptionValue(highs, "parallel", "off");
  Highs_setIntOptionValue(highs, "simplex_scale_strategy", scale_strategy);
  Highs_setIntOptionValue(highs, "simplex_dual_edge_weight_strategy",
                          edge_weight_strategy);
  Highs_setDoubleOptionValue(highs, "dual_simplex_cost_perturbation_multiplier",
                             cost_perturbation);
  Highs_setIntOptionValue(highs, "random_seed", 0);
  if (iteration_limit > 0)
    Highs_setIntOptionValue(highs, "simplex_iteration_limit", iteration_limit);
  if (dse_weight_error_threshold >= 0.0)
    Highs_setDoubleOptionValue(
        highs, "dual_steepest_edge_weight_log_error_threshold",
        dse_weight_error_threshold);
  if (primal_edge_weight_strategy >= 0)
    Highs_setIntOptionValue(highs, "simplex_primal_edge_weight_strategy",
                            primal_edge_weight_strategy);
  if (primal_bound_perturbation >= 0.0)
    Highs_setDoubleOptionValue(highs,
                               "primal_simplex_bound_perturbation_multiplier",
                               primal_bound_perturbation);
  if (simplex_strategy >= 0)
    Highs_setIntOptionValue(highs, "simplex_strategy", simplex_strategy);
  if (analysis_level > 0) {
    Highs_setIntOptionValue(highs, "highs_analysis_level", analysis_level);
    // Trace par itération (objectif dual mis à jour) : exige le journal dev.
    Highs_setBoolOptionValue(highs, "output_flag", 1);
    Highs_setIntOptionValue(highs, "log_dev_level", 2);
  }
  HighsInt status = Highs_passLp(
      highs, num_col, num_row, num_nz, kHighsMatrixFormatColwise, sense,
      offset, col_cost.data(), col_lower.data(), col_upper.data(),
      row_lower.data(), row_upper.data(), a_start.data(), a_index.data(),
      a_value.data());
  if (status == kHighsStatusError) {
    printf("passlp -1 0 0\n");
    Highs_destroy(highs);
    return 2;
  }

  // Boucle d'opérations.
  char op[64];
  long long solve_index = 0;
  while (fscanf(stdin, "%63s", op) == 1) {
    if (strcmp(op, "end") == 0) {
      break;
    } else if (strcmp(op, "solve") == 0) {
      const int fd = analysis_level > 0 ? -1 : silence_stdout();
      status = Highs_run(highs);
      HighsInt model_status = Highs_getModelStatus(highs);
      double objective = Highs_getObjectiveValue(highs);
      HighsInt iterations = 0;
      Highs_getIntInfoValue(highs, "simplex_iteration_count", &iterations);
      HighsInt num_primal_infeasibilities = -1;
      Highs_getIntInfoValue(highs, "num_primal_infeasibilities",
                            &num_primal_infeasibilities);
      HighsInt num_dual_infeasibilities = -1;
      Highs_getIntInfoValue(highs, "num_dual_infeasibilities",
                            &num_dual_infeasibilities);
      std::vector<double> col_value(num_col), col_dual(num_col),
          row_value(num_row), row_dual(num_row);
      if (print_vectors)
        Highs_getSolution(highs, col_value.data(), col_dual.data(),
                          row_value.data(), row_dual.data());
      restore_stdout(fd);
      solve_index++;
      printf("solve %lld %lld %.17g %lld %lld %lld\n", solve_index,
             (long long)model_status, objective, (long long)iterations,
             (long long)num_primal_infeasibilities,
             (long long)num_dual_infeasibilities);
      if (print_vectors) {
        printf("colvalue");
        for (long long j = 0; j < num_col; j++) printf(" %a", col_value[j]);
        printf("\ncoldual");
        for (long long j = 0; j < num_col; j++) printf(" %a", col_dual[j]);
        printf("\nrowvalue");
        for (long long i = 0; i < num_row; i++)
          printf(" %a", row_value[i]);
        printf("\nrowdual");
        for (long long i = 0; i < num_row; i++) printf(" %a", row_dual[i]);
        printf("\n");
        // Base HiGHS (codes : 0 lower, 1 basic, 2 upper, 3 zero, 4 nonbasic).
        std::vector<HighsInt> col_status(num_col), row_status(num_row);
        if (Highs_getBasis(highs, col_status.data(), row_status.data()) ==
            kHighsStatusOk) {
          printf("basiscol");
          for (long long j = 0; j < num_col; j++)
            printf(" %d", (int)col_status[j]);
          printf("\nbasisrow");
          for (long long i = 0; i < num_row; i++)
            printf(" %d", (int)row_status[i]);
          printf("\n");
        }
      }
    } else if (strcmp(op, "change_col_bounds") == 0) {
      long long c;
      double lo, up;
      if (fscanf(stdin, "%lld %lf %lf", &c, &lo, &up) != 3) return 2;
      Highs_changeColBounds(highs, (HighsInt)(c - 1), lo, up);
    } else if (strcmp(op, "change_cols_bounds") == 0) {
      long long n;
      if (fscanf(stdin, "%lld", &n) != 1) return 2;
      for (long long k = 0; k < n; k++) {
        long long c;
        double lo, up;
        if (fscanf(stdin, "%lld %lf %lf", &c, &lo, &up) != 3) return 2;
        Highs_changeColBounds(highs, (HighsInt)(c - 1), lo, up);
      }
    } else if (strcmp(op, "change_row_bounds") == 0) {
      long long r;
      double lo, up;
      if (fscanf(stdin, "%lld %lf %lf", &r, &lo, &up) != 3) return 2;
      Highs_changeRowBounds(highs, (HighsInt)(r - 1), lo, up);
    } else if (strcmp(op, "change_rows_bounds") == 0) {
      long long n;
      if (fscanf(stdin, "%lld", &n) != 1) return 2;
      for (long long k = 0; k < n; k++) {
        long long r;
        double lo, up;
        if (fscanf(stdin, "%lld %lf %lf", &r, &lo, &up) != 3) return 2;
        Highs_changeRowBounds(highs, (HighsInt)(r - 1), lo, up);
      }
    } else if (strcmp(op, "change_col_cost") == 0) {
      long long c;
      double value;
      if (fscanf(stdin, "%lld %lf", &c, &value) != 2) return 2;
      Highs_changeColCost(highs, (HighsInt)(c - 1), value);
    } else if (strcmp(op, "change_cols_cost") == 0) {
      long long n;
      if (fscanf(stdin, "%lld", &n) != 1) return 2;
      for (long long k = 0; k < n; k++) {
        long long c;
        double value;
        if (fscanf(stdin, "%lld %lf", &c, &value) != 2) return 2;
        Highs_changeColCost(highs, (HighsInt)(c - 1), value);
      }
    } else if (strcmp(op, "change_coeff") == 0) {
      long long r, c;
      double value;
      if (fscanf(stdin, "%lld %lld %lf", &r, &c, &value) != 3) return 2;
      Highs_changeCoeff(highs, (HighsInt)(r - 1), (HighsInt)(c - 1), value);
    } else if (strcmp(op, "change_sense") == 0) {
      long long s;
      if (fscanf(stdin, "%lld", &s) != 1) return 2;
      Highs_changeObjectiveSense(highs, (HighsInt)s);
    } else if (strcmp(op, "limit") == 0) {
      // Limite d'itérations du prochain solve (comparaison pas à pas).
      long long n;
      if (fscanf(stdin, "%lld", &n) != 1) return 2;
      Highs_setIntOptionValue(highs, "simplex_iteration_limit", (HighsInt)n);
    } else {
      fprintf(stderr, "sequence_oracle : opération inconnue « %s »\n", op);
      Highs_destroy(highs);
      return 2;
    }
  }
  Highs_destroy(highs);
  return 0;
}
