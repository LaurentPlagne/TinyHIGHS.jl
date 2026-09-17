// Oracle « solveur complet » pour M3b : résout un LP avec HiGHS (libhighs de la
// source gelée) dans la configuration qui correspond à la tranche portée —
// simplexe dual Dantzig, sans presolve, sans perturbation de coûts — et imprime
// statut, objectif et nombre d'itérations.
//
//   julia_simplex/oracle/build.sh
//
// Entrée (stdin), cas séparés par rien de plus que les lignes du format :
//   num_col num_row nnz
//   col_cost[0..n-1] (%a)
//   col_lower ; col_upper ; row_lower ; row_upper (%a)
//   a_start[0..n] (0-based) ; a_index (0-based) ; a_value (%a)
//   sense offset (%a)
// Sortie : une ligne par cas `status objective iterations`.
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char** argv) {
  // Configuration optionnelle : scale edge_weight perturb (défauts de M3b :
  // sans échelle, Dantzig, sans perturbation de coûts).
  const HighsInt scale_strategy = argc > 1 ? (HighsInt)atoi(argv[1]) : 0;
  const HighsInt edge_weight_strategy = argc > 2 ? (HighsInt)atoi(argv[2]) : 0;
  const double cost_perturbation = argc > 3 ? atof(argv[3]) : 0.0;
  // Limite d'itérations optionnelle (0 = aucune) : permet de comparer l'état
  // des deux solveurs pas à pas.
  const HighsInt iteration_limit = argc > 4 ? (HighsInt)atoi(argv[4]) : 0;
  // Seuil de bascule DSE → Devex (5e argument) ; 0 le rend immédiat.
  const double dse_weight_error_threshold = argc > 5 ? atof(argv[5]) : -1.0;
  // 6e : stratégie de poids primale (-1 choose, 0 Dantzig, 1 Devex, 2 PSE).
  const HighsInt primal_edge_weight_strategy =
      argc > 6 ? (HighsInt)atoi(argv[6]) : -1;
  // 7e : multiplicateur de perturbation des bornes primal ; < 0 = défaut.
  const double primal_bound_perturbation = argc > 7 ? atof(argv[7]) : -1.0;
  // 8e : simplex_strategy (4 = primal) ; -1 = défaut (dual).
  const HighsInt simplex_strategy = argc > 8 ? (HighsInt)atoi(argv[8]) : -1;
  for (;;) {
    long long num_col, num_row, num_nz;
    if (fscanf(stdin, "%lld %lld %lld", &num_col, &num_row, &num_nz) != 3)
      break;
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
    HighsInt status = Highs_passLp(
        highs, num_col, num_row, num_nz, kHighsMatrixFormatColwise, sense,
        offset, col_cost.data(), col_lower.data(), col_upper.data(),
        row_lower.data(), row_upper.data(), a_start.data(), a_index.data(),
        a_value.data());
    if (status == kHighsStatusError) {
      printf("-1 0 0\n");
      Highs_destroy(highs);
      continue;
    }
    // La source gelée force le contrôle `debugPrimalSteepestEdgeWeights` en
    // stratégie primale steepest edge, qui écrit sur stdout via printf (le
    // reste du journal reste muet avec output_flag = 0). Détourner stdout
    // pendant le solve : les tests ne lisent que les résultats.
    fflush(stdout);
    const int stdout_fd = dup(fileno(stdout));
    if (stdout_fd >= 0) {
      FILE* sink = freopen("/dev/null", "w", stdout);
      (void)sink;
    }
    status = Highs_run(highs);
    HighsInt model_status = Highs_getModelStatus(highs);
    double objective = Highs_getObjectiveValue(highs);
    HighsInt iterations = 0;
    Highs_getIntInfoValue(highs, "simplex_iteration_count", &iterations);
    fflush(stdout);
    if (stdout_fd >= 0) {
      dup2(stdout_fd, fileno(stdout));
      close(stdout_fd);
    }
    printf("%lld %.17g %lld\n", (long long)model_status, objective,
           (long long)iterations);
    // Valeurs finales (%a) : colvalue, coldual, rowvalue, rowdual. Comparées
    // bit-à-bit par le test lorsque le statut est optimal.
    std::vector<double> col_value(num_col), col_dual(num_col),
        row_value(num_row), row_dual(num_row);
    Highs_getSolution(highs, col_value.data(), col_dual.data(),
                      row_value.data(), row_dual.data());
    printf("colvalue");
    for (long long j = 0; j < num_col; j++) printf(" %a", col_value[j]);
    printf("\ncoldual");
    for (long long j = 0; j < num_col; j++) printf(" %a", col_dual[j]);
    printf("\nrowvalue");
    for (long long i = 0; i < num_row; i++) printf(" %a", row_value[i]);
    printf("\nrowdual");
    for (long long i = 0; i < num_row; i++) printf(" %a", row_dual[i]);
    printf("\n");
    Highs_destroy(highs);
  }
  return 0;
}
