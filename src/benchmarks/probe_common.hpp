#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bench {

#ifndef BENCH_OMP_CHUNK
#define BENCH_OMP_CHUNK 16
#endif

struct Timer {
    std::chrono::steady_clock::time_point t0;

    Timer() : t0(std::chrono::steady_clock::now()) {}

    double elapsed_ms() const {
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

struct CsrMatrix {
    int rows = 0;
    int cols = 0;
    long long nnz = 0;
    bool symmetric = false;
    std::vector<int> rowptr;
    std::vector<int> colidx;
};

struct BasicStats {
    double avg = 0.0;
    int max = 0;
    int min = 0;
    double stddev = 0.0;
    double cv = 0.0;
    double skewness = 0.0;
    double empty_ratio = 0.0;
};

struct TileProbe {
    int tile_m = 0;
    int tile_n = 0;
    int tile_rows = 0;
    int tile_cols = 0;
    long long numtile = 0;
    double tile_density = 0.0;
    double tile_sparsity = 0.0;
    BasicStats nnz_per_tile;
    BasicStats tile_row_nnz;
    BasicStats tile_col_nnz;
    double tile_fill_avg = 0.0;
    double tile_fill_max = 0.0;
    BasicStats tiles_per_row;
    BasicStats tiles_per_col;
    int hist_1 = 0;
    int hist_2_4 = 0;
    int hist_4_8 = 0;
    int hist_8_16 = 0;
    int hist_16_32 = 0;
    int hist_32_64 = 0;
    int hist_64_128 = 0;
    int hist_128_plus = 0;
};

struct TileMatrix {
    int tile_m = 0;
    int tile_n = 0;
    int tile_rows = 0;
    int tile_cols = 0;
    std::vector<std::vector<int>> rows;
};

struct CMatchedStats {
    int tile_m = 0;
    long long numblkC = 0;
    double avg_matchedcnt = 0.0;
    int max_matchedcnt = 0;
};

struct RunningIntStats {
    long long count = 0;
    long double sum = 0.0;
    long double sumsq = 0.0;
    int max = 0;
    int min = std::numeric_limits<int>::max();
    long long empty = 0;

    void add(int v) {
        count++;
        sum += static_cast<long double>(v);
        sumsq += static_cast<long double>(v) * static_cast<long double>(v);
        max = std::max(max, v);
        min = std::min(min, v);
        empty += (v == 0);
    }

    void add_zeros(long long n) {
        if (n <= 0) {
            return;
        }
        count += n;
        min = std::min(min, 0);
        empty += n;
    }

    void merge(const RunningIntStats &other) {
        if (other.count == 0) {
            return;
        }
        count += other.count;
        sum += other.sum;
        sumsq += other.sumsq;
        max = std::max(max, other.max);
        min = std::min(min, other.min);
        empty += other.empty;
    }

    BasicStats finish() const {
        BasicStats s;
        if (count == 0) {
            return s;
        }
        s.avg = static_cast<double>(sum / static_cast<long double>(count));
        const long double mean = sum / static_cast<long double>(count);
        long double var = sumsq / static_cast<long double>(count) - mean * mean;
        if (var < 0.0L) {
            var = 0.0L;
        }
        s.stddev = std::sqrt(static_cast<double>(var));
        s.cv = (s.avg != 0.0) ? s.stddev / s.avg : 0.0;
        s.max = max;
        s.min = min;
        s.empty_ratio = static_cast<double>(empty) / static_cast<double>(count);
        return s;
    }
};

inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

inline bool is_power_of_two(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

inline int log2_power_of_two(int x) {
    if (x <= 0 || (x & (x - 1)) != 0) {
        throw std::runtime_error("tile size must be a power of two");
    }
    int shift = 0;
    while ((1 << shift) < x) {
        shift++;
    }
    return shift;
}

inline std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline bool parse_size_line(std::ifstream &in, int *rows, int *cols, long long *nnz) {
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        std::istringstream iss(line);
        if (iss >> *rows >> *cols >> *nnz) {
            return true;
        }
    }
    return false;
}

inline CsrMatrix load_matrix_market(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open matrix file: " + path);
    }

    std::string banner;
    if (!std::getline(in, banner)) {
        throw std::runtime_error("empty matrix file: " + path);
    }

    const std::string banner_l = lower_copy(banner);
    const bool symmetric = banner_l.find("symmetric") != std::string::npos ||
                           banner_l.find("hermitian") != std::string::npos;
    const bool pattern = banner_l.find("pattern") != std::string::npos;

    int rows = 0;
    int cols = 0;
    long long reported_nnz = 0;
    if (!parse_size_line(in, &rows, &cols, &reported_nnz)) {
        throw std::runtime_error("could not parse matrix size: " + path);
    }

    std::vector<std::pair<int, int>> coords;
    coords.reserve(static_cast<size_t>(reported_nnz * (symmetric ? 2 : 1)));

    std::string line;
    long long read_entries = 0;
    while (read_entries < reported_nnz && std::getline(in, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        std::istringstream iss(line);
        int r = 0;
        int c = 0;
        if (!(iss >> r >> c)) {
            continue;
        }
        (void)pattern;
        --r;
        --c;
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw std::runtime_error("matrix index out of range: " + path);
        }
        coords.emplace_back(r, c);
        if (symmetric && r != c) {
            coords.emplace_back(c, r);
        }
        ++read_entries;
    }

    std::sort(coords.begin(), coords.end());

    CsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = static_cast<long long>(coords.size());
    matrix.symmetric = symmetric;
    matrix.rowptr.assign(static_cast<size_t>(rows + 1), 0);
    matrix.colidx.resize(coords.size());

    for (const auto &rc : coords) {
        matrix.rowptr[static_cast<size_t>(rc.first + 1)]++;
    }
    for (int i = 0; i < rows; ++i) {
        matrix.rowptr[static_cast<size_t>(i + 1)] += matrix.rowptr[static_cast<size_t>(i)];
    }

    std::vector<int> cursor = matrix.rowptr;
    for (const auto &rc : coords) {
        const int pos = cursor[static_cast<size_t>(rc.first)]++;
        matrix.colidx[static_cast<size_t>(pos)] = rc.second;
    }
    return matrix;
}

inline CsrMatrix transpose(const CsrMatrix &a) {
    CsrMatrix t;
    t.rows = a.cols;
    t.cols = a.rows;
    t.nnz = a.nnz;
    t.symmetric = a.symmetric;
    t.rowptr.assign(static_cast<size_t>(t.rows + 1), 0);
    t.colidx.assign(static_cast<size_t>(a.nnz), 0);

    for (int row = 0; row < a.rows; ++row) {
        for (int p = a.rowptr[static_cast<size_t>(row)]; p < a.rowptr[static_cast<size_t>(row + 1)]; ++p) {
            t.rowptr[static_cast<size_t>(a.colidx[static_cast<size_t>(p)] + 1)]++;
        }
    }
    for (int i = 0; i < t.rows; ++i) {
        t.rowptr[static_cast<size_t>(i + 1)] += t.rowptr[static_cast<size_t>(i)];
    }

    std::vector<int> cursor = t.rowptr;
    for (int row = 0; row < a.rows; ++row) {
        for (int p = a.rowptr[static_cast<size_t>(row)]; p < a.rowptr[static_cast<size_t>(row + 1)]; ++p) {
            const int col = a.colidx[static_cast<size_t>(p)];
            const int dst = cursor[static_cast<size_t>(col)]++;
            t.colidx[static_cast<size_t>(dst)] = row;
        }
    }
    return t;
}

inline TileProbe probe_tile_size(const CsrMatrix &a, int tile_m, int tile_n) {
    if (tile_m > 32 || tile_n > 32) {
        throw std::runtime_error("probe_tile_size supports tile dimensions up to 32");
    }

    struct TileAccum {
        int tile_col = 0;
        int nnz = 0;
        std::array<int, 32> col_nnz{};
    };
    struct Partial {
        long long numtile = 0;
        long double sparsity_sum = 0.0;
        double tile_fill_max = 0.0;
        RunningIntStats row_nnz_stats;
        RunningIntStats col_nnz_stats;
        int hist_1 = 0;
        int hist_2_4 = 0;
        int hist_4_8 = 0;
        int hist_8_16 = 0;
        int hist_16_32 = 0;
        int hist_32_64 = 0;
        int hist_64_128 = 0;
        int hist_128_plus = 0;
    };

    TileProbe p;
    p.tile_m = tile_m;
    p.tile_n = tile_n;
    p.tile_rows = ceil_div(a.rows, tile_m);
    p.tile_cols = ceil_div(a.cols, tile_n);
    const int tile_n_shift = log2_power_of_two(tile_n);
    const int tile_n_mask = tile_n - 1;

    const int max_threads = std::max(1, omp_get_max_threads());
    std::vector<Partial> partials(static_cast<size_t>(max_threads));

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        Partial &part = partials[static_cast<size_t>(thread_id)];
        std::vector<int> slot_of_col(static_cast<size_t>(p.tile_cols), -1);
        std::vector<int> touched_cols;
        std::vector<TileAccum> active_tiles;
        touched_cols.reserve(128);
        active_tiles.reserve(128);

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int tr = 0; tr < p.tile_rows; ++tr) {
            active_tiles.clear();
            touched_cols.clear();
            const int row_begin = tr * tile_m;
            const int row_end = std::min(a.rows, row_begin + tile_m);
            const int actual_rows = row_end - row_begin;
            long long row_slots = 0;
            long long nonzero_row_slots = 0;

            for (int r = row_begin; r < row_end; ++r) {
                int current_tc = -1;
                int row_tile_nnz = 0;
                for (int pos = a.rowptr[static_cast<size_t>(r)]; pos < a.rowptr[static_cast<size_t>(r + 1)]; ++pos) {
                    const int col = a.colidx[static_cast<size_t>(pos)];
                    const int tc = col >> tile_n_shift;
                    const int local_c = col & tile_n_mask;
                    int slot = slot_of_col[static_cast<size_t>(tc)];
                    if (slot < 0) {
                        slot = static_cast<int>(active_tiles.size());
                        slot_of_col[static_cast<size_t>(tc)] = slot;
                        touched_cols.push_back(tc);
                        TileAccum acc;
                        acc.tile_col = tc;
                        active_tiles.push_back(acc);
                    }

                    TileAccum &acc = active_tiles[static_cast<size_t>(slot)];
                    acc.nnz++;
                    acc.col_nnz[static_cast<size_t>(local_c)]++;

                    if (tc != current_tc) {
                        if (current_tc >= 0) {
                            part.row_nnz_stats.add(row_tile_nnz);
                            nonzero_row_slots++;
                        }
                        current_tc = tc;
                        row_tile_nnz = 1;
                    } else {
                        row_tile_nnz++;
                    }
                }
                if (current_tc >= 0) {
                    part.row_nnz_stats.add(row_tile_nnz);
                    nonzero_row_slots++;
                }
            }

            for (const TileAccum &acc : active_tiles) {
                const int actual_cols = std::min(tile_n, a.cols - acc.tile_col * tile_n);
                const int tile_area = actual_rows * actual_cols;
                part.numtile++;
                row_slots += actual_rows;

                if (tile_area > 0) {
                    const double fill = static_cast<double>(acc.nnz) / static_cast<double>(tile_area);
                    part.sparsity_sum += std::max(0.0, 1.0 - fill);
                    part.tile_fill_max = std::max(part.tile_fill_max, fill);
                }
                for (int i = 0; i < actual_cols; ++i) {
                    part.col_nnz_stats.add(acc.col_nnz[static_cast<size_t>(i)]);
                }

                if (acc.nnz == 1) part.hist_1++;
                else if (acc.nnz < 4) part.hist_2_4++;
                else if (acc.nnz < 8) part.hist_4_8++;
                else if (acc.nnz < 16) part.hist_8_16++;
                else if (acc.nnz < 32) part.hist_16_32++;
                else if (acc.nnz < 64) part.hist_32_64++;
                else if (acc.nnz < 128) part.hist_64_128++;
                else part.hist_128_plus++;
            }

            part.row_nnz_stats.add_zeros(row_slots - nonzero_row_slots);
            for (int tc : touched_cols) {
                slot_of_col[static_cast<size_t>(tc)] = -1;
            }
        }
    }

    long double sparsity_sum = 0.0;
    RunningIntStats row_nnz_stats;
    RunningIntStats col_nnz_stats;
    for (const Partial &part : partials) {
        p.numtile += part.numtile;
        sparsity_sum += part.sparsity_sum;
        p.tile_fill_max = std::max(p.tile_fill_max, part.tile_fill_max);
        row_nnz_stats.merge(part.row_nnz_stats);
        col_nnz_stats.merge(part.col_nnz_stats);
        p.hist_1 += part.hist_1;
        p.hist_2_4 += part.hist_2_4;
        p.hist_4_8 += part.hist_4_8;
        p.hist_8_16 += part.hist_8_16;
        p.hist_16_32 += part.hist_16_32;
        p.hist_32_64 += part.hist_32_64;
        p.hist_64_128 += part.hist_64_128;
        p.hist_128_plus += part.hist_128_plus;
    }

    p.tile_density = (p.tile_rows && p.tile_cols)
                         ? static_cast<double>(p.numtile) / static_cast<double>(p.tile_rows) / static_cast<double>(p.tile_cols)
                         : 0.0;
    p.tile_sparsity = p.numtile ? static_cast<double>(sparsity_sum / static_cast<long double>(p.numtile)) : 0.0;
    p.tile_fill_avg = 1.0 - p.tile_sparsity;
    p.tile_row_nnz = row_nnz_stats.finish();
    p.tile_col_nnz = col_nnz_stats.finish();
    return p;
}

inline std::vector<TileProbe> probe_tile_sizes_parallel(
    const CsrMatrix &a,
    const std::vector<std::pair<int, int>> &tile_sizes) {
    struct TileAccum {
        int tile_col = 0;
        int nnz = 0;
        std::array<int, 32> col_nnz{};
    };
    struct Scratch {
        std::vector<int> slot_of_col;
        std::vector<int> touched_cols;
        std::vector<TileAccum> active_tiles;

        void ensure(int tile_cols) {
            if (slot_of_col.size() != static_cast<size_t>(tile_cols)) {
                slot_of_col.assign(static_cast<size_t>(tile_cols), -1);
                touched_cols.reserve(128);
                active_tiles.reserve(128);
            }
        }
    };
    struct Partial {
        long long numtile = 0;
        long double sparsity_sum = 0.0;
        double tile_fill_max = 0.0;
        RunningIntStats row_nnz_stats;
        RunningIntStats col_nnz_stats;
        int hist_1 = 0;
        int hist_2_4 = 0;
        int hist_4_8 = 0;
        int hist_8_16 = 0;
        int hist_16_32 = 0;
        int hist_32_64 = 0;
        int hist_64_128 = 0;
        int hist_128_plus = 0;
    };

    std::vector<TileProbe> probes(tile_sizes.size());
    std::vector<std::pair<int, int>> tasks;
    std::vector<int> tile_n_shifts(tile_sizes.size());
    for (size_t i = 0; i < tile_sizes.size(); ++i) {
        const int tile_m = tile_sizes[i].first;
        const int tile_n = tile_sizes[i].second;
        if (tile_m > 32 || tile_n > 32) {
            throw std::runtime_error("probe_tile_sizes_parallel supports tile dimensions up to 32");
        }
        TileProbe &p = probes[i];
        p.tile_m = tile_m;
        p.tile_n = tile_n;
        p.tile_rows = ceil_div(a.rows, tile_m);
        p.tile_cols = ceil_div(a.cols, tile_n);
        tile_n_shifts[i] = log2_power_of_two(tile_n);
        for (int tr = 0; tr < p.tile_rows; ++tr) {
            tasks.emplace_back(static_cast<int>(i), tr);
        }
    }

    const int max_threads = std::max(1, omp_get_max_threads());
    std::vector<Partial> partials(tile_sizes.size() * static_cast<size_t>(max_threads));

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        std::vector<Scratch> scratch(tile_sizes.size());

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int task_id = 0; task_id < static_cast<int>(tasks.size()); ++task_id) {
            const int shape = tasks[static_cast<size_t>(task_id)].first;
            const int tr = tasks[static_cast<size_t>(task_id)].second;
            const TileProbe &p = probes[static_cast<size_t>(shape)];
            const int tile_m = p.tile_m;
            const int tile_n = p.tile_n;
            const int tile_n_shift = tile_n_shifts[static_cast<size_t>(shape)];
            const int tile_n_mask = tile_n - 1;
            Scratch &ws = scratch[static_cast<size_t>(shape)];
            Partial &part = partials[static_cast<size_t>(shape) * static_cast<size_t>(max_threads) +
                                     static_cast<size_t>(thread_id)];
            ws.ensure(p.tile_cols);
            ws.active_tiles.clear();
            ws.touched_cols.clear();

            const int row_begin = tr * tile_m;
            const int row_end = std::min(a.rows, row_begin + tile_m);
            const int actual_rows = row_end - row_begin;
            long long row_slots = 0;
            long long nonzero_row_slots = 0;

            for (int r = row_begin; r < row_end; ++r) {
                int current_tc = -1;
                int row_tile_nnz = 0;
                for (int pos = a.rowptr[static_cast<size_t>(r)]; pos < a.rowptr[static_cast<size_t>(r + 1)]; ++pos) {
                    const int col = a.colidx[static_cast<size_t>(pos)];
                    const int tc = col >> tile_n_shift;
                    const int local_c = col & tile_n_mask;
                    int slot = ws.slot_of_col[static_cast<size_t>(tc)];
                    if (slot < 0) {
                        slot = static_cast<int>(ws.active_tiles.size());
                        ws.slot_of_col[static_cast<size_t>(tc)] = slot;
                        ws.touched_cols.push_back(tc);
                        TileAccum acc;
                        acc.tile_col = tc;
                        ws.active_tiles.push_back(acc);
                    }

                    TileAccum &acc = ws.active_tiles[static_cast<size_t>(slot)];
                    acc.nnz++;
                    acc.col_nnz[static_cast<size_t>(local_c)]++;

                    if (tc != current_tc) {
                        if (current_tc >= 0) {
                            part.row_nnz_stats.add(row_tile_nnz);
                            nonzero_row_slots++;
                        }
                        current_tc = tc;
                        row_tile_nnz = 1;
                    } else {
                        row_tile_nnz++;
                    }
                }
                if (current_tc >= 0) {
                    part.row_nnz_stats.add(row_tile_nnz);
                    nonzero_row_slots++;
                }
            }

            for (const TileAccum &acc : ws.active_tiles) {
                const int actual_cols = std::min(tile_n, a.cols - acc.tile_col * tile_n);
                const int tile_area = actual_rows * actual_cols;
                part.numtile++;
                row_slots += actual_rows;

                if (tile_area > 0) {
                    const double fill = static_cast<double>(acc.nnz) / static_cast<double>(tile_area);
                    part.sparsity_sum += std::max(0.0, 1.0 - fill);
                    part.tile_fill_max = std::max(part.tile_fill_max, fill);
                }
                for (int i = 0; i < actual_cols; ++i) {
                    part.col_nnz_stats.add(acc.col_nnz[static_cast<size_t>(i)]);
                }

                if (acc.nnz == 1) part.hist_1++;
                else if (acc.nnz < 4) part.hist_2_4++;
                else if (acc.nnz < 8) part.hist_4_8++;
                else if (acc.nnz < 16) part.hist_8_16++;
                else if (acc.nnz < 32) part.hist_16_32++;
                else if (acc.nnz < 64) part.hist_32_64++;
                else if (acc.nnz < 128) part.hist_64_128++;
                else part.hist_128_plus++;
            }

            part.row_nnz_stats.add_zeros(row_slots - nonzero_row_slots);
            for (int tc : ws.touched_cols) {
                ws.slot_of_col[static_cast<size_t>(tc)] = -1;
            }
        }
    }

    for (size_t shape = 0; shape < probes.size(); ++shape) {
        TileProbe &p = probes[shape];
        long double sparsity_sum = 0.0;
        RunningIntStats row_nnz_stats;
        RunningIntStats col_nnz_stats;
        for (int tid = 0; tid < max_threads; ++tid) {
            const Partial &part = partials[shape * static_cast<size_t>(max_threads) + static_cast<size_t>(tid)];
            p.numtile += part.numtile;
            sparsity_sum += part.sparsity_sum;
            p.tile_fill_max = std::max(p.tile_fill_max, part.tile_fill_max);
            row_nnz_stats.merge(part.row_nnz_stats);
            col_nnz_stats.merge(part.col_nnz_stats);
            p.hist_1 += part.hist_1;
            p.hist_2_4 += part.hist_2_4;
            p.hist_4_8 += part.hist_4_8;
            p.hist_8_16 += part.hist_8_16;
            p.hist_16_32 += part.hist_16_32;
            p.hist_32_64 += part.hist_32_64;
            p.hist_64_128 += part.hist_64_128;
            p.hist_128_plus += part.hist_128_plus;
        }
        p.tile_density = (p.tile_rows && p.tile_cols)
                             ? static_cast<double>(p.numtile) / static_cast<double>(p.tile_rows) /
                                   static_cast<double>(p.tile_cols)
                             : 0.0;
        p.tile_sparsity = p.numtile ? static_cast<double>(sparsity_sum / static_cast<long double>(p.numtile)) : 0.0;
        p.tile_fill_avg = 1.0 - p.tile_sparsity;
        p.tile_row_nnz = row_nnz_stats.finish();
        p.tile_col_nnz = col_nnz_stats.finish();
    }
    return probes;
}

struct BaseTileEntry8 {
    int tile_row = 0;
    int tile_col = 0;
    int nnz = 0;
    std::array<int, 8> row_nnz{};
    std::array<int, 8> col_nnz{};
};

inline std::vector<std::vector<BaseTileEntry8>> build_base_tile_entries_8(const CsrMatrix &a) {
    const int base = 8;
    const int base_tile_rows = ceil_div(a.rows, base);
    const int base_tile_cols = ceil_div(a.cols, base);
    std::vector<std::vector<BaseTileEntry8>> rows(static_cast<size_t>(base_tile_rows));

#pragma omp parallel
    {
        std::vector<int> slot_of_col(static_cast<size_t>(base_tile_cols), -1);
        std::vector<int> touched_cols;
        std::vector<BaseTileEntry8> active_tiles;
        touched_cols.reserve(128);
        active_tiles.reserve(128);

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int tr = 0; tr < base_tile_rows; ++tr) {
            touched_cols.clear();
            active_tiles.clear();

            const int row_begin = tr * base;
            const int row_end = std::min(a.rows, row_begin + base);
            for (int r = row_begin; r < row_end; ++r) {
                const int local_r = r & (base - 1);
                for (int pos = a.rowptr[static_cast<size_t>(r)]; pos < a.rowptr[static_cast<size_t>(r + 1)]; ++pos) {
                    const int col = a.colidx[static_cast<size_t>(pos)];
                    const int tc = col >> 3;
                    const int local_c = col & (base - 1);
                    int slot = slot_of_col[static_cast<size_t>(tc)];
                    if (slot < 0) {
                        slot = static_cast<int>(active_tiles.size());
                        slot_of_col[static_cast<size_t>(tc)] = slot;
                        touched_cols.push_back(tc);
                        BaseTileEntry8 acc;
                        acc.tile_row = tr;
                        acc.tile_col = tc;
                        active_tiles.push_back(acc);
                    }

                    BaseTileEntry8 &acc = active_tiles[static_cast<size_t>(slot)];
                    acc.nnz++;
                    acc.row_nnz[static_cast<size_t>(local_r)]++;
                    acc.col_nnz[static_cast<size_t>(local_c)]++;
                }
            }

            rows[static_cast<size_t>(tr)] = active_tiles;
            for (int tc : touched_cols) {
                slot_of_col[static_cast<size_t>(tc)] = -1;
            }
        }
    }

    return rows;
}

inline bool can_probe_by_merging_base8(const std::vector<std::pair<int, int>> &tile_sizes) {
    for (const auto &tile_size : tile_sizes) {
        const int tile_m = tile_size.first;
        const int tile_n = tile_size.second;
        if (tile_m < 8 || tile_n < 8 || tile_m > 32 || tile_n > 32 ||
            (tile_m % 8) != 0 || (tile_n % 8) != 0 ||
            !is_power_of_two(tile_m) || !is_power_of_two(tile_n)) {
            return false;
        }
    }
    return true;
}

inline std::vector<TileProbe> probe_tile_sizes_merged_power2(
    const CsrMatrix &a,
    const std::vector<std::pair<int, int>> &tile_sizes) {
    if (!can_probe_by_merging_base8(tile_sizes)) {
        return probe_tile_sizes_parallel(a, tile_sizes);
    }

    const int base = 8;
    struct TileAccum {
        int tile_col = 0;
        int nnz = 0;
        std::array<int, 32> row_nnz{};
        std::array<int, 32> col_nnz{};
    };
    struct Scratch {
        std::vector<int> slot_of_col;
        std::vector<int> touched_cols;
        std::vector<TileAccum> active_tiles;

        void ensure(int tile_cols) {
            if (slot_of_col.size() != static_cast<size_t>(tile_cols)) {
                slot_of_col.assign(static_cast<size_t>(tile_cols), -1);
                touched_cols.reserve(128);
                active_tiles.reserve(128);
            }
        }
    };
    struct Partial {
        long long numtile = 0;
        long double sparsity_sum = 0.0;
        double tile_fill_max = 0.0;
        RunningIntStats row_nnz_stats;
        RunningIntStats col_nnz_stats;
        int hist_1 = 0;
        int hist_2_4 = 0;
        int hist_4_8 = 0;
        int hist_8_16 = 0;
        int hist_16_32 = 0;
        int hist_32_64 = 0;
        int hist_64_128 = 0;
        int hist_128_plus = 0;
    };

    std::vector<TileProbe> probes(tile_sizes.size());
    std::vector<std::pair<int, int>> tasks;
    std::vector<int> row_factors(tile_sizes.size());
    std::vector<int> col_factors(tile_sizes.size());

    for (size_t i = 0; i < tile_sizes.size(); ++i) {
        const int tile_m = tile_sizes[i].first;
        const int tile_n = tile_sizes[i].second;
        TileProbe &p = probes[i];
        p.tile_m = tile_m;
        p.tile_n = tile_n;
        p.tile_rows = ceil_div(a.rows, tile_m);
        p.tile_cols = ceil_div(a.cols, tile_n);
        row_factors[i] = tile_m / base;
        col_factors[i] = tile_n / base;
        for (int tr = 0; tr < p.tile_rows; ++tr) {
            tasks.emplace_back(static_cast<int>(i), tr);
        }
    }

    const int base_tile_rows = ceil_div(a.rows, base);
    const int base_tile_cols = ceil_div(a.cols, base);
    std::vector<std::vector<BaseTileEntry8>> base_rows(static_cast<size_t>(base_tile_rows));
    const int max_threads = std::max(1, omp_get_max_threads());
    std::vector<Partial> partials(tile_sizes.size() * static_cast<size_t>(max_threads));

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        std::vector<int> base_slot_of_col(static_cast<size_t>(base_tile_cols), -1);
        std::vector<int> base_touched_cols;
        std::vector<BaseTileEntry8> base_active_tiles;
        base_touched_cols.reserve(128);
        base_active_tiles.reserve(128);

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int base_tr = 0; base_tr < base_tile_rows; ++base_tr) {
            base_touched_cols.clear();
            base_active_tiles.clear();

            const int row_begin = base_tr * base;
            const int row_end = std::min(a.rows, row_begin + base);
            for (int r = row_begin; r < row_end; ++r) {
                const int local_r = r & (base - 1);
                for (int pos = a.rowptr[static_cast<size_t>(r)]; pos < a.rowptr[static_cast<size_t>(r + 1)]; ++pos) {
                    const int col = a.colidx[static_cast<size_t>(pos)];
                    const int tc = col >> 3;
                    const int local_c = col & (base - 1);
                    int slot = base_slot_of_col[static_cast<size_t>(tc)];
                    if (slot < 0) {
                        slot = static_cast<int>(base_active_tiles.size());
                        base_slot_of_col[static_cast<size_t>(tc)] = slot;
                        base_touched_cols.push_back(tc);
                        BaseTileEntry8 acc;
                        acc.tile_row = base_tr;
                        acc.tile_col = tc;
                        base_active_tiles.push_back(acc);
                    }

                    BaseTileEntry8 &acc = base_active_tiles[static_cast<size_t>(slot)];
                    acc.nnz++;
                    acc.row_nnz[static_cast<size_t>(local_r)]++;
                    acc.col_nnz[static_cast<size_t>(local_c)]++;
                }
            }

            base_rows[static_cast<size_t>(base_tr)] = base_active_tiles;
            for (int tc : base_touched_cols) {
                base_slot_of_col[static_cast<size_t>(tc)] = -1;
            }
        }

        std::vector<Scratch> scratch(tile_sizes.size());

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int task_id = 0; task_id < static_cast<int>(tasks.size()); ++task_id) {
            const int shape = tasks[static_cast<size_t>(task_id)].first;
            const int tr = tasks[static_cast<size_t>(task_id)].second;
            const TileProbe &p = probes[static_cast<size_t>(shape)];
            const int tile_m = p.tile_m;
            const int tile_n = p.tile_n;
            const int row_factor = row_factors[static_cast<size_t>(shape)];
            const int col_factor = col_factors[static_cast<size_t>(shape)];
            const int base_row_begin = tr * row_factor;
            const int base_row_end = std::min(static_cast<int>(base_rows.size()), base_row_begin + row_factor);
            const int row_begin = tr * tile_m;
            const int row_end = std::min(a.rows, row_begin + tile_m);
            const int actual_rows = row_end - row_begin;

            Scratch &ws = scratch[static_cast<size_t>(shape)];
            Partial &part = partials[static_cast<size_t>(shape) * static_cast<size_t>(max_threads) +
                                     static_cast<size_t>(thread_id)];
            ws.ensure(p.tile_cols);
            ws.touched_cols.clear();
            ws.active_tiles.clear();

            for (int base_tr = base_row_begin; base_tr < base_row_end; ++base_tr) {
                const int row_offset = (base_tr - base_row_begin) * base;
                for (const BaseTileEntry8 &base_tile : base_rows[static_cast<size_t>(base_tr)]) {
                    const int tc = base_tile.tile_col / col_factor;
                    const int col_offset = (base_tile.tile_col % col_factor) * base;
                    int slot = ws.slot_of_col[static_cast<size_t>(tc)];
                    if (slot < 0) {
                        slot = static_cast<int>(ws.active_tiles.size());
                        ws.slot_of_col[static_cast<size_t>(tc)] = slot;
                        ws.touched_cols.push_back(tc);
                        TileAccum acc;
                        acc.tile_col = tc;
                        ws.active_tiles.push_back(acc);
                    }

                    TileAccum &acc = ws.active_tiles[static_cast<size_t>(slot)];
                    acc.nnz += base_tile.nnz;
                    for (int i = 0; i < base; ++i) {
                        acc.row_nnz[static_cast<size_t>(row_offset + i)] +=
                            base_tile.row_nnz[static_cast<size_t>(i)];
                        acc.col_nnz[static_cast<size_t>(col_offset + i)] +=
                            base_tile.col_nnz[static_cast<size_t>(i)];
                    }
                }
            }

            for (const TileAccum &acc : ws.active_tiles) {
                const int actual_cols = std::min(tile_n, a.cols - acc.tile_col * tile_n);
                const int tile_area = actual_rows * actual_cols;
                part.numtile++;

                if (tile_area > 0) {
                    const double fill = static_cast<double>(acc.nnz) / static_cast<double>(tile_area);
                    part.sparsity_sum += std::max(0.0, 1.0 - fill);
                    part.tile_fill_max = std::max(part.tile_fill_max, fill);
                }
                for (int i = 0; i < actual_rows; ++i) {
                    part.row_nnz_stats.add(acc.row_nnz[static_cast<size_t>(i)]);
                }
                for (int i = 0; i < actual_cols; ++i) {
                    part.col_nnz_stats.add(acc.col_nnz[static_cast<size_t>(i)]);
                }

                if (acc.nnz == 1) part.hist_1++;
                else if (acc.nnz < 4) part.hist_2_4++;
                else if (acc.nnz < 8) part.hist_4_8++;
                else if (acc.nnz < 16) part.hist_8_16++;
                else if (acc.nnz < 32) part.hist_16_32++;
                else if (acc.nnz < 64) part.hist_32_64++;
                else if (acc.nnz < 128) part.hist_64_128++;
                else part.hist_128_plus++;
            }

            for (int tc : ws.touched_cols) {
                ws.slot_of_col[static_cast<size_t>(tc)] = -1;
            }
        }
    }

    for (size_t shape = 0; shape < probes.size(); ++shape) {
        TileProbe &p = probes[shape];
        long double sparsity_sum = 0.0;
        RunningIntStats row_nnz_stats;
        RunningIntStats col_nnz_stats;
        for (int tid = 0; tid < max_threads; ++tid) {
            const Partial &part = partials[shape * static_cast<size_t>(max_threads) + static_cast<size_t>(tid)];
            p.numtile += part.numtile;
            sparsity_sum += part.sparsity_sum;
            p.tile_fill_max = std::max(p.tile_fill_max, part.tile_fill_max);
            row_nnz_stats.merge(part.row_nnz_stats);
            col_nnz_stats.merge(part.col_nnz_stats);
            p.hist_1 += part.hist_1;
            p.hist_2_4 += part.hist_2_4;
            p.hist_4_8 += part.hist_4_8;
            p.hist_8_16 += part.hist_8_16;
            p.hist_16_32 += part.hist_16_32;
            p.hist_32_64 += part.hist_32_64;
            p.hist_64_128 += part.hist_64_128;
            p.hist_128_plus += part.hist_128_plus;
        }
        p.tile_density = (p.tile_rows && p.tile_cols)
                             ? static_cast<double>(p.numtile) / static_cast<double>(p.tile_rows) /
                                   static_cast<double>(p.tile_cols)
                             : 0.0;
        p.tile_sparsity = p.numtile ? static_cast<double>(sparsity_sum / static_cast<long double>(p.numtile)) : 0.0;
        p.tile_fill_avg = 1.0 - p.tile_sparsity;
        p.tile_row_nnz = row_nnz_stats.finish();
        p.tile_col_nnz = col_nnz_stats.finish();
    }

    return probes;
}

inline TileMatrix build_tile_matrix(const CsrMatrix &a, int tile_m, int tile_n) {
    TileMatrix t;
    t.tile_m = tile_m;
    t.tile_n = tile_n;
    t.tile_rows = ceil_div(a.rows, tile_m);
    t.tile_cols = ceil_div(a.cols, tile_n);
    const int tile_n_shift = log2_power_of_two(tile_n);

    t.rows.assign(static_cast<size_t>(t.tile_rows), {});

#pragma omp parallel
    {
        std::vector<int> slot_of_col(static_cast<size_t>(t.tile_cols), -1);
        std::vector<int> touched_cols;
        touched_cols.reserve(128);

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int tr = 0; tr < t.tile_rows; ++tr) {
            touched_cols.clear();
            const int row_begin = tr * tile_m;
            const int row_end = std::min(a.rows, row_begin + tile_m);
            for (int r = row_begin; r < row_end; ++r) {
                for (int pos = a.rowptr[static_cast<size_t>(r)]; pos < a.rowptr[static_cast<size_t>(r + 1)]; ++pos) {
                    const int tc = a.colidx[static_cast<size_t>(pos)] >> tile_n_shift;
                    if (slot_of_col[static_cast<size_t>(tc)] < 0) {
                        slot_of_col[static_cast<size_t>(tc)] = 1;
                        touched_cols.push_back(tc);
                    }
                }
            }
            t.rows[static_cast<size_t>(tr)] = touched_cols;
            for (int tc : touched_cols) {
                slot_of_col[static_cast<size_t>(tc)] = -1;
            }
        }
    }
    return t;
}

inline std::vector<TileMatrix> build_tile_matrices_parallel(const CsrMatrix &a, const std::vector<int> &tile_ms) {
    struct Scratch {
        std::vector<int> slot_of_col;
        std::vector<int> touched_cols;

        void ensure(int tile_cols) {
            if (slot_of_col.size() != static_cast<size_t>(tile_cols)) {
                slot_of_col.assign(static_cast<size_t>(tile_cols), -1);
                touched_cols.reserve(128);
            }
        }
    };

    std::vector<TileMatrix> tiles(tile_ms.size());
    std::vector<std::pair<int, int>> tasks;
    std::vector<int> tile_shifts(tile_ms.size());
    for (size_t i = 0; i < tile_ms.size(); ++i) {
        const int tm = tile_ms[i];
        TileMatrix &t = tiles[i];
        t.tile_m = tm;
        t.tile_n = tm;
        t.tile_rows = ceil_div(a.rows, tm);
        t.tile_cols = ceil_div(a.cols, tm);
        tile_shifts[i] = log2_power_of_two(tm);
        t.rows.assign(static_cast<size_t>(t.tile_rows), {});
        for (int tr = 0; tr < t.tile_rows; ++tr) {
            tasks.emplace_back(static_cast<int>(i), tr);
        }
    }

#pragma omp parallel
    {
        std::vector<Scratch> scratch(tile_ms.size());

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int task_id = 0; task_id < static_cast<int>(tasks.size()); ++task_id) {
            const int shape = tasks[static_cast<size_t>(task_id)].first;
            const int tr = tasks[static_cast<size_t>(task_id)].second;
            TileMatrix &t = tiles[static_cast<size_t>(shape)];
            const int tile_shift = tile_shifts[static_cast<size_t>(shape)];
            Scratch &ws = scratch[static_cast<size_t>(shape)];
            ws.ensure(t.tile_cols);
            ws.touched_cols.clear();

            const int row_begin = tr * t.tile_m;
            const int row_end = std::min(a.rows, row_begin + t.tile_m);
            for (int r = row_begin; r < row_end; ++r) {
                for (int pos = a.rowptr[static_cast<size_t>(r)]; pos < a.rowptr[static_cast<size_t>(r + 1)]; ++pos) {
                    const int tc = a.colidx[static_cast<size_t>(pos)] >> tile_shift;
                    if (ws.slot_of_col[static_cast<size_t>(tc)] < 0) {
                        ws.slot_of_col[static_cast<size_t>(tc)] = 1;
                        ws.touched_cols.push_back(tc);
                    }
                }
            }
            t.rows[static_cast<size_t>(tr)] = ws.touched_cols;
            for (int tc : ws.touched_cols) {
                ws.slot_of_col[static_cast<size_t>(tc)] = -1;
            }
        }
    }
    return tiles;
}

inline TileMatrix merge_tile_matrix_power2(const TileMatrix &base, int tile_m) {
    if (tile_m == base.tile_m && tile_m == base.tile_n) {
        return base;
    }
    if (tile_m < base.tile_m || tile_m < base.tile_n ||
        (tile_m % base.tile_m) != 0 || (tile_m % base.tile_n) != 0 ||
        !is_power_of_two(tile_m)) {
        throw std::runtime_error("cannot merge tile matrix to requested power-of-two tile size");
    }

    const int row_factor = tile_m / base.tile_m;
    const int col_factor = tile_m / base.tile_n;

    TileMatrix merged;
    merged.tile_m = tile_m;
    merged.tile_n = tile_m;
    merged.tile_rows = ceil_div(base.tile_rows, row_factor);
    merged.tile_cols = ceil_div(base.tile_cols, col_factor);
    merged.rows.assign(static_cast<size_t>(merged.tile_rows), {});

#pragma omp parallel
    {
        std::vector<int> marker(static_cast<size_t>(merged.tile_cols), -1);
        std::vector<int> touched_cols;
        touched_cols.reserve(128);

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int tr = 0; tr < merged.tile_rows; ++tr) {
            touched_cols.clear();
            const int base_row_begin = tr * row_factor;
            const int base_row_end = std::min(base.tile_rows, base_row_begin + row_factor);
            for (int base_tr = base_row_begin; base_tr < base_row_end; ++base_tr) {
                for (const int base_tc : base.rows[static_cast<size_t>(base_tr)]) {
                    const int tc = base_tc / col_factor;
                    if (marker[static_cast<size_t>(tc)] < 0) {
                        marker[static_cast<size_t>(tc)] = 1;
                        touched_cols.push_back(tc);
                    }
                }
            }
            merged.rows[static_cast<size_t>(tr)] = touched_cols;
            for (int tc : touched_cols) {
                marker[static_cast<size_t>(tc)] = -1;
            }
        }
    }

    return merged;
}

inline bool can_build_tile_matrices_by_merging(const std::vector<int> &tile_ms) {
    if (tile_ms.empty()) {
        return true;
    }
    int base_tile = tile_ms[0];
    for (const int tile_m : tile_ms) {
        if (tile_m <= 0 || !is_power_of_two(tile_m)) {
            return false;
        }
        base_tile = std::min(base_tile, tile_m);
    }
    for (const int tile_m : tile_ms) {
        if ((tile_m % base_tile) != 0) {
            return false;
        }
    }
    return true;
}

inline std::vector<TileMatrix> build_tile_matrices_merged_power2(const CsrMatrix &a, const std::vector<int> &tile_ms) {
    if (!can_build_tile_matrices_by_merging(tile_ms)) {
        return build_tile_matrices_parallel(a, tile_ms);
    }
    if (tile_ms.empty()) {
        return {};
    }

    int base_tile = tile_ms[0];
    for (const int tile_m : tile_ms) {
        base_tile = std::min(base_tile, tile_m);
    }

    TileMatrix base;
    base.tile_m = base_tile;
    base.tile_n = base_tile;
    base.tile_rows = ceil_div(a.rows, base_tile);
    base.tile_cols = ceil_div(a.cols, base_tile);
    base.rows.assign(static_cast<size_t>(base.tile_rows), {});

    std::vector<TileMatrix> tiles(tile_ms.size());
    std::vector<std::pair<int, int>> merge_tasks;
    std::vector<int> row_factors(tile_ms.size(), 1);
    std::vector<int> col_factors(tile_ms.size(), 1);
    for (size_t i = 0; i < tile_ms.size(); ++i) {
        const int tile_m = tile_ms[i];
        if (tile_m == base_tile) {
            continue;
        }
        TileMatrix &t = tiles[i];
        t.tile_m = tile_m;
        t.tile_n = tile_m;
        row_factors[i] = tile_m / base_tile;
        col_factors[i] = tile_m / base_tile;
        t.tile_rows = ceil_div(base.tile_rows, row_factors[i]);
        t.tile_cols = ceil_div(base.tile_cols, col_factors[i]);
        t.rows.assign(static_cast<size_t>(t.tile_rows), {});
        for (int tr = 0; tr < t.tile_rows; ++tr) {
            merge_tasks.emplace_back(static_cast<int>(i), tr);
        }
    }

    const int base_shift = log2_power_of_two(base_tile);

#pragma omp parallel
    {
        std::vector<int> base_marker(static_cast<size_t>(base.tile_cols), -1);
        std::vector<int> touched_cols;
        touched_cols.reserve(128);

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int tr = 0; tr < base.tile_rows; ++tr) {
            touched_cols.clear();
            const int row_begin = tr * base_tile;
            const int row_end = std::min(a.rows, row_begin + base_tile);
            for (int r = row_begin; r < row_end; ++r) {
                for (int pos = a.rowptr[static_cast<size_t>(r)]; pos < a.rowptr[static_cast<size_t>(r + 1)]; ++pos) {
                    const int tc = a.colidx[static_cast<size_t>(pos)] >> base_shift;
                    if (base_marker[static_cast<size_t>(tc)] < 0) {
                        base_marker[static_cast<size_t>(tc)] = 1;
                        touched_cols.push_back(tc);
                    }
                }
            }
            base.rows[static_cast<size_t>(tr)] = touched_cols;
            for (int tc : touched_cols) {
                base_marker[static_cast<size_t>(tc)] = -1;
            }
        }

        std::vector<int> merge_marker;
        std::vector<int> merge_touched_cols;
        merge_touched_cols.reserve(128);

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int task_id = 0; task_id < static_cast<int>(merge_tasks.size()); ++task_id) {
            const int shape = merge_tasks[static_cast<size_t>(task_id)].first;
            const int tr = merge_tasks[static_cast<size_t>(task_id)].second;
            TileMatrix &t = tiles[static_cast<size_t>(shape)];
            if (merge_marker.size() != static_cast<size_t>(t.tile_cols)) {
                merge_marker.assign(static_cast<size_t>(t.tile_cols), -1);
            }
            merge_touched_cols.clear();

            const int row_factor = row_factors[static_cast<size_t>(shape)];
            const int col_factor = col_factors[static_cast<size_t>(shape)];
            const int base_row_begin = tr * row_factor;
            const int base_row_end = std::min(base.tile_rows, base_row_begin + row_factor);
            for (int base_tr = base_row_begin; base_tr < base_row_end; ++base_tr) {
                for (const int base_tc : base.rows[static_cast<size_t>(base_tr)]) {
                    const int tc = base_tc / col_factor;
                    if (merge_marker[static_cast<size_t>(tc)] < 0) {
                        merge_marker[static_cast<size_t>(tc)] = 1;
                        merge_touched_cols.push_back(tc);
                    }
                }
            }
            t.rows[static_cast<size_t>(tr)] = merge_touched_cols;
            for (int tc : merge_touched_cols) {
                merge_marker[static_cast<size_t>(tc)] = -1;
            }
        }
    }

    for (size_t i = 0; i < tile_ms.size(); ++i) {
        if (tile_ms[i] == base_tile) {
            tiles[i] = base;
        }
    }

    return tiles;
}

inline CMatchedStats compute_c_matched_stats(const TileMatrix &a, const TileMatrix &b, int tile_m) {
    if (a.tile_cols != b.tile_rows) {
        throw std::runtime_error("tile dimensions are incompatible for C probe");
    }

    CMatchedStats s;
    s.tile_m = tile_m;

    struct Partial {
        long long numblkC = 0;
        long long matched_sum = 0;
        int max_matchedcnt = 0;
    };
    const int max_threads = std::max(1, omp_get_max_threads());
    std::vector<Partial> partials(static_cast<size_t>(max_threads));

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        Partial &part = partials[static_cast<size_t>(thread_id)];
        std::vector<int> marker(static_cast<size_t>(b.tile_cols), -1);
        std::vector<int> counts(static_cast<size_t>(b.tile_cols), 0);
        std::vector<int> touched_cols;
        touched_cols.reserve(128);

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int i = 0; i < a.tile_rows; ++i) {
            touched_cols.clear();
            for (const int k : a.rows[static_cast<size_t>(i)]) {
                if (k < 0 || k >= b.tile_rows) {
                    continue;
                }
                for (const int j : b.rows[static_cast<size_t>(k)]) {
                    if (marker[static_cast<size_t>(j)] != i) {
                        marker[static_cast<size_t>(j)] = i;
                        counts[static_cast<size_t>(j)] = 1;
                        touched_cols.push_back(j);
                    } else {
                        counts[static_cast<size_t>(j)]++;
                    }
                }
            }

            for (const int j : touched_cols) {
                const int matched = counts[static_cast<size_t>(j)];
                part.numblkC++;
                part.matched_sum += matched;
                part.max_matchedcnt = std::max(part.max_matchedcnt, matched);
            }
        }
    }

    long long matched_sum = 0;
    for (const Partial &part : partials) {
        s.numblkC += part.numblkC;
        matched_sum += part.matched_sum;
        s.max_matchedcnt = std::max(s.max_matchedcnt, part.max_matchedcnt);
    }
    s.avg_matchedcnt = s.numblkC ? static_cast<double>(matched_sum) / static_cast<double>(s.numblkC) : 0.0;
    return s;
}

inline std::vector<CMatchedStats> compute_c_matched_stats_parallel(
    const std::vector<TileMatrix> &a_tiles,
    const std::vector<TileMatrix> &b_tiles,
    const std::vector<int> &tile_ms) {
    struct Scratch {
        std::vector<int> marker;
        std::vector<int> counts;
        std::vector<int> touched_cols;

        void ensure(int tile_cols) {
            if (marker.size() != static_cast<size_t>(tile_cols)) {
                marker.assign(static_cast<size_t>(tile_cols), -1);
                counts.assign(static_cast<size_t>(tile_cols), 0);
                touched_cols.reserve(128);
            }
        }
    };
    struct Partial {
        long long numblkC = 0;
        long long matched_sum = 0;
        int max_matchedcnt = 0;
    };

    if (a_tiles.size() != b_tiles.size() || a_tiles.size() != tile_ms.size()) {
        throw std::runtime_error("batched C matched stats got mismatched tile vector sizes");
    }

    std::vector<CMatchedStats> stats(tile_ms.size());
    std::vector<std::pair<int, int>> tasks;
    for (size_t shape = 0; shape < tile_ms.size(); ++shape) {
        if (a_tiles[shape].tile_cols != b_tiles[shape].tile_rows) {
            throw std::runtime_error("tile dimensions are incompatible for batched C probe");
        }
        stats[shape].tile_m = tile_ms[shape];
        for (int i = 0; i < a_tiles[shape].tile_rows; ++i) {
            tasks.emplace_back(static_cast<int>(shape), i);
        }
    }

    const int max_threads = std::max(1, omp_get_max_threads());
    std::vector<Partial> partials(tile_ms.size() * static_cast<size_t>(max_threads));

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        std::vector<Scratch> scratch(tile_ms.size());

#pragma omp for schedule(dynamic, BENCH_OMP_CHUNK)
        for (int task_id = 0; task_id < static_cast<int>(tasks.size()); ++task_id) {
            const int shape = tasks[static_cast<size_t>(task_id)].first;
            const int i = tasks[static_cast<size_t>(task_id)].second;
            const TileMatrix &a = a_tiles[static_cast<size_t>(shape)];
            const TileMatrix &b = b_tiles[static_cast<size_t>(shape)];
            Scratch &ws = scratch[static_cast<size_t>(shape)];
            Partial &part = partials[static_cast<size_t>(shape) * static_cast<size_t>(max_threads) +
                                     static_cast<size_t>(thread_id)];
            ws.ensure(b.tile_cols);
            ws.touched_cols.clear();

            for (const int k : a.rows[static_cast<size_t>(i)]) {
                if (k < 0 || k >= b.tile_rows) {
                    continue;
                }
                for (const int j : b.rows[static_cast<size_t>(k)]) {
                    if (ws.marker[static_cast<size_t>(j)] != i) {
                        ws.marker[static_cast<size_t>(j)] = i;
                        ws.counts[static_cast<size_t>(j)] = 1;
                        ws.touched_cols.push_back(j);
                    } else {
                        ws.counts[static_cast<size_t>(j)]++;
                    }
                }
            }
            for (const int j : ws.touched_cols) {
                const int matched = ws.counts[static_cast<size_t>(j)];
                part.numblkC++;
                part.matched_sum += matched;
                part.max_matchedcnt = std::max(part.max_matchedcnt, matched);
            }
        }
    }

    for (size_t shape = 0; shape < tile_ms.size(); ++shape) {
        long long matched_sum = 0;
        CMatchedStats &s = stats[shape];
        for (int tid = 0; tid < max_threads; ++tid) {
            const Partial &part = partials[shape * static_cast<size_t>(max_threads) + static_cast<size_t>(tid)];
            s.numblkC += part.numblkC;
            matched_sum += part.matched_sum;
            s.max_matchedcnt = std::max(s.max_matchedcnt, part.max_matchedcnt);
        }
        s.avg_matchedcnt = s.numblkC ? static_cast<double>(matched_sum) / static_cast<double>(s.numblkC) : 0.0;
    }
    return stats;
}

inline std::string basename(const std::string &path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

}  // namespace bench
