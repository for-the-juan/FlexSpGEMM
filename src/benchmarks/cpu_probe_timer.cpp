#include "probe_common.hpp"

#include <omp.h>

namespace {

const int kATileSizes[][2] = {
    {8, 8}, {16, 8}, {8, 16}, {16, 16}, {16, 32},
    {32, 16}, {32, 32}, {8, 32}, {32, 8},
};

const int kCTileMs[] = {8, 16, 32};

struct ProbeResult {
    std::string matrix_name;
    int rows = 0;
    int cols = 0;
    long long nnz = 0;
    bool symmetric = false;
    double load_ms = 0.0;
    double a_probe_ms = 0.0;
    double c_build_ms = 0.0;
    double c_feature_ms = 0.0;
    std::vector<bench::TileProbe> a_probes;
    std::vector<bench::CMatchedStats> c_stats;
};

void usage(const char *prog) {
    std::fprintf(stderr, "Usage: %s [--csv] [--aat] <A.mtx> [B.mtx]\n", prog);
}

std::vector<std::pair<int, int>> a_tile_sizes() {
    std::vector<std::pair<int, int>> sizes;
    sizes.reserve(sizeof(kATileSizes) / sizeof(kATileSizes[0]));
    for (const auto &tile_size : kATileSizes) {
        sizes.emplace_back(tile_size[0], tile_size[1]);
    }
    return sizes;
}

std::vector<int> c_tile_sizes() {
    return std::vector<int>(kCTileMs, kCTileMs + sizeof(kCTileMs) / sizeof(kCTileMs[0]));
}

bool use_merged_a_probe(const char **selected_impl) {
    const char *env_impl = std::getenv("BENCH_A_PROBE_IMPL");
    if (env_impl) {
        if (std::strcmp(env_impl, "merge") == 0) {
            *selected_impl = "merge";
            return true;
        }
        if (std::strcmp(env_impl, "direct") == 0) {
            *selected_impl = "direct";
            return false;
        }
    }

    *selected_impl = "default-merge";
    return true;
}

bool use_merged_c_build(const char **selected_impl) {
    const char *env_impl = std::getenv("BENCH_C_PROBE_BUILD_IMPL");
    if (env_impl && std::strcmp(env_impl, "merge") == 0) {
        *selected_impl = "merge";
        return true;
    }

    *selected_impl = "direct";
    return false;
}

const bench::TileProbe *find_a_probe(const std::vector<bench::TileProbe> &probes, int tile_m, int tile_n) {
    for (const auto &probe : probes) {
        if (probe.tile_m == tile_m && probe.tile_n == tile_n) {
            return &probe;
        }
    }
    return nullptr;
}

const bench::CMatchedStats *find_c_stats(const std::vector<bench::CMatchedStats> &rows, int tile_m) {
    for (const auto &row : rows) {
        if (row.tile_m == tile_m) {
            return &row;
        }
    }
    return nullptr;
}

void print_csv_header() {
    std::printf("matrix,rows,cols,nnz,symmetric,load_ms,"
                "cpu_a_probe_ms,cpu_a_tiles_16x16,"
                "cpu_c_build_ms,cpu_c_feature_ms,cpu_c_total_ms,cpu_probe_total_ms,"
                "cpu_c_tiles_16,cpu_c_avg_16,cpu_c_max_16\n");
}

void print_csv_row(const ProbeResult &result) {
    const bench::TileProbe *a16 = find_a_probe(result.a_probes, 16, 16);
    const bench::CMatchedStats *c16 = find_c_stats(result.c_stats, 16);
    const double c_total_ms = result.c_build_ms + result.c_feature_ms;
    const double probe_total_ms = result.a_probe_ms + c_total_ms;

    std::printf("%s,%d,%d,%lld,%d,%.6f,%.6f,%lld,%.6f,%.6f,%.6f,%.6f,%lld,%.6f,%d\n",
                result.matrix_name.c_str(),
                result.rows,
                result.cols,
                result.nnz,
                result.symmetric ? 1 : 0,
                result.load_ms,
                result.a_probe_ms,
                a16 ? a16->numtile : 0,
                result.c_build_ms,
                result.c_feature_ms,
                c_total_ms,
                probe_total_ms,
                c16 ? c16->numblkC : 0,
                c16 ? c16->avg_matchedcnt : 0.0,
                c16 ? c16->max_matchedcnt : 0);
}

void print_human_summary(const ProbeResult &result,
                         const char *a_impl,
                         const char *c_impl,
                         bool has_b_path,
                         bool aat) {
    const bench::TileProbe *a16 = find_a_probe(result.a_probes, 16, 16);
    const bench::CMatchedStats *c16 = find_c_stats(result.c_stats, 16);
    const double c_total_ms = result.c_build_ms + result.c_feature_ms;
    const double probe_total_ms = result.a_probe_ms + c_total_ms;

    std::printf("============================================================\n");
    std::printf("  Standalone CPU A/C Tile Feature Probe\n");
    std::printf("============================================================\n");
    std::printf("Matrix: %s\n", result.matrix_name.c_str());
    std::printf("Rows: %d, Cols: %d, nnz: %lld, Symmetric: %s\n",
                result.rows, result.cols, result.nnz, result.symmetric ? "Yes" : "No");
    std::printf("C mode: %s\n", has_b_path ? "A * B" : (aat ? "A * A^T" : "A * A"));
    std::printf("OpenMP threads: %d\n", omp_get_max_threads());
    std::printf("Load time: %.2f ms\n", result.load_ms);
    std::printf("A probe implementation: %s\n", a_impl);
    std::printf("C tile build implementation: %s\n", c_impl);
    std::printf("A probe time: %.2f ms\n", result.a_probe_ms);
    std::printf("C build time: %.2f ms\n", result.c_build_ms);
    std::printf("C feature time: %.2f ms\n", result.c_feature_ms);
    std::printf("C total time: %.2f ms\n", c_total_ms);
    std::printf("Probe total time: %.2f ms\n", probe_total_ms);
    if (a16) {
        std::printf("A tile 16x16: tile_num=%lld, tile_sparsity=%.6f\n",
                    a16->numtile, a16->tile_sparsity);
    }
    if (c16) {
        std::printf("C tile 16x16: tile_num=%lld, avg_matched=%.2f, max_matched=%d\n",
                    c16->numblkC, c16->avg_matchedcnt, c16->max_matchedcnt);
    }
}

ProbeResult run_probe(const std::string &a_path,
                      const std::string &b_path,
                      bool has_b_path,
                      bool aat,
                      const char **a_impl,
                      const char **c_impl) {
    bench::Timer load_timer;
    bench::CsrMatrix a = bench::load_matrix_market(a_path);
    bench::CsrMatrix b_storage;
    const bench::CsrMatrix *b = &a;
    if (has_b_path) {
        b_storage = bench::load_matrix_market(b_path);
        b = &b_storage;
    } else if (aat) {
        b_storage = bench::transpose(a);
        b = &b_storage;
    }
    const double load_ms = load_timer.elapsed_ms();

    if (a.cols != b->rows) {
        throw std::runtime_error("A cols and B rows do not match");
    }
    if (!aat && !has_b_path && a.rows != a.cols) {
        throw std::runtime_error("matrix must be square for C = A*A");
    }

    const bool merge_a = use_merged_a_probe(a_impl);
    const bool merge_c = use_merged_c_build(c_impl);
    const bool same_ab = !has_b_path && !aat;

    bench::Timer a_timer;
    std::vector<bench::TileProbe> a_probes = merge_a
                                                 ? bench::probe_tile_sizes_merged_power2(a, a_tile_sizes())
                                                 : bench::probe_tile_sizes_parallel(a, a_tile_sizes());
    const double a_probe_ms = a_timer.elapsed_ms();

    bench::Timer c_build_timer;
    const std::vector<int> tile_ms = c_tile_sizes();
    std::vector<bench::TileMatrix> tiles_a = merge_c
                                                 ? bench::build_tile_matrices_merged_power2(a, tile_ms)
                                                 : bench::build_tile_matrices_parallel(a, tile_ms);
    std::vector<bench::TileMatrix> tiles_b;
    if (!same_ab) {
        tiles_b = merge_c
                      ? bench::build_tile_matrices_merged_power2(*b, tile_ms)
                      : bench::build_tile_matrices_parallel(*b, tile_ms);
    }
    const double c_build_ms = c_build_timer.elapsed_ms();

    bench::Timer c_feature_timer;
    const std::vector<bench::TileMatrix> &tiles_b_ref = same_ab ? tiles_a : tiles_b;
    std::vector<bench::CMatchedStats> c_stats =
        bench::compute_c_matched_stats_parallel(tiles_a, tiles_b_ref, tile_ms);
    const double c_feature_ms = c_feature_timer.elapsed_ms();

    ProbeResult result;
    result.matrix_name = bench::basename(a_path);
    result.rows = a.rows;
    result.cols = a.cols;
    result.nnz = a.nnz;
    result.symmetric = a.symmetric;
    result.load_ms = load_ms;
    result.a_probe_ms = a_probe_ms;
    result.c_build_ms = c_build_ms;
    result.c_feature_ms = c_feature_ms;
    result.a_probes = std::move(a_probes);
    result.c_stats = std::move(c_stats);
    return result;
}

}  // namespace

int main(int argc, char **argv) {
    bool csv = false;
    bool aat = false;
    std::vector<std::string> paths;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--csv") {
            csv = true;
        } else if (arg == "--aat") {
            aat = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            paths.push_back(arg);
        }
    }

    if (paths.empty() || paths.size() > 2) {
        usage(argv[0]);
        return 1;
    }

    try {
        const bool has_b_path = paths.size() == 2;
        const char *a_impl = nullptr;
        const char *c_impl = nullptr;
        ProbeResult result = run_probe(paths[0],
                                       has_b_path ? paths[1] : std::string(),
                                       has_b_path,
                                       aat,
                                       &a_impl,
                                       &c_impl);

        if (csv) {
            print_csv_header();
            print_csv_row(result);
        } else {
            print_human_summary(result, a_impl, c_impl, has_b_path, aat);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
