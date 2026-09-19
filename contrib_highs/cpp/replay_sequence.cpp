// ==============================================================================
// Standalone C++ Sequence Replay Benchmark for HiGHS
// Zero Julia dependency - Pure C++11 using official Highs API
// ==============================================================================

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Highs.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <base.lp> <operations.txt> [repeats=1]" << std::endl;
        return 1;
    }

    const std::string base_lp_path = argv[1];
    const std::string ops_path = argv[2];
    const int repeats = (argc > 3) ? std::atoi(argv[3]) : 1;

    // Read operations file into memory once
    std::ifstream ops_file(ops_path);
    if (!ops_file.is_open()) {
        std::cerr << "Error: Cannot open operations file: " << ops_path << std::endl;
        return 1;
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ops_file, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    ops_file.close();

    std::cout << "================================================================================" << std::endl;
    std::cout << " HiGHS C++ Native Sequence Benchmark (Warm-Start)" << std::endl;
    std::cout << " Base model : " << base_lp_path << std::endl;
    std::cout << " Operations : " << ops_path << " (" << lines.size() << " instructions)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    double best_time_ms = 1e9;
    int total_solves = 0;
    int total_iters = 0;
    double final_obj = 0.0;
    int final_status = -1;

    for (int rep = 0; rep < repeats; rep++) {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("solver", "simplex");
        highs.setOptionValue("presolve", "off");
        highs.setOptionValue("parallel", "off");
        // Match TinyHiGHS and the frozen sequence oracle: no scaling and the
        // same deterministic dual-simplex defaults. Keeping these explicit is
        // essential for separating numerical pivot effects from configuration
        // differences in the warm-start replay.
        highs.setOptionValue("simplex_scale_strategy", 0);
        highs.setOptionValue("simplex_dual_edge_weight_strategy", -1);
        highs.setOptionValue("dual_simplex_cost_perturbation_multiplier", 1.0);
        highs.setOptionValue("random_seed", 0);

        HighsStatus status = highs.readModel(base_lp_path);
        if (status != HighsStatus::kOk) {
            std::cerr << "Error reading base LP model: " << base_lp_path << std::endl;
            return 1;
        }

        int current_solves = 0;
        int current_iters = 0;
        int current_status = -1;

        auto t0 = std::chrono::high_resolution_clock::now();

        for (const auto& l : lines) {
            std::istringstream iss(l);
            std::string op;
            iss >> op;

            if (op == "change_col_bounds") {
                HighsInt col;
                double lo, up;
                iss >> col >> lo >> up;
                highs.changeColBounds(col - 1, lo, up);
            } else if (op == "change_cols_bounds") {
                int n;
                iss >> n;
                for (int i = 0; i < n; i++) {
                    HighsInt col;
                    double lo, up;
                    iss >> col >> lo >> up;
                    highs.changeColBounds(col - 1, lo, up);
                }
            } else if (op == "change_row_bounds") {
                HighsInt row;
                double lo, up;
                iss >> row >> lo >> up;
                highs.changeRowBounds(row - 1, lo, up);
            } else if (op == "change_rows_bounds") {
                int n;
                iss >> n;
                for (int i = 0; i < n; i++) {
                    HighsInt row;
                    double lo, up;
                    iss >> row >> lo >> up;
                    highs.changeRowBounds(row - 1, lo, up);
                }
            } else if (op == "change_cols_cost") {
                int n;
                iss >> n;
                for (int i = 0; i < n; i++) {
                    HighsInt col;
                    double val;
                    iss >> col >> val;
                    highs.changeColCost(col - 1, val);
                }
            } else if (op == "solve") {
                highs.run();
                current_solves++;
                current_iters += highs.getInfo().simplex_iteration_count;
                final_obj = highs.getInfo().objective_function_value;
                current_status = static_cast<int>(highs.getModelStatus());
            } else if (op == "end") {
                break;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (elapsed_ms < best_time_ms) {
            best_time_ms = elapsed_ms;
        }
        total_solves = current_solves;
        total_iters = current_iters;
        final_status = current_status;
    }

    double us_per_solve = (best_time_ms * 1000.0) / total_solves;

    std::printf(" Total Solves      : %d\n", total_solves);
    std::printf(" Total Iterations  : %d\n", total_iters);
    std::printf(" Final Objective   : %.10g\n", final_obj);
    std::printf(" Final Status Code : %d\n", final_status);
    std::printf(" Best Elapsed Time : %.2f ms\n", best_time_ms);
    std::printf(" Average Solve Time: %.1f µs / solve\n", us_per_solve);
    std::cout << "================================================================================" << std::endl;

    return 0;
}
