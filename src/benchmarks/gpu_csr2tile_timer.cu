#include "probe_common.hpp"

#include "../csr2tile.h"
#include "../gpu_csr2tile.h"

#include <cuda_runtime.h>

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <regex>
#include <vector>

namespace {

struct BatchSpec {
    std::string matrix_name;
    std::string matrix_path;
    int tile_m;
    int tile_n;
};

std::string trim_whitespace(std::string_view value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    std::string result(value.substr(start, end - start));
    if (result.size() >= 2 && result.front() == '\"' && result.back() == '\"') {
        result = result.substr(1, result.size() - 2);
    }
    return result;
}

std::vector<std::string> split_csv_row(const std::string &line) {
    std::vector<std::string> columns;
    std::string current;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '\"') {
            in_quotes = !in_quotes;
            current.push_back(ch);
            continue;
        }
        if (ch == ',' && !in_quotes) {
            columns.push_back(trim_whitespace(current));
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    columns.push_back(trim_whitespace(current));
    return columns;
}

bool parse_tile_from_log_path(const std::string &path, int &tile_m, int &tile_n) {
    static const std::regex kLogTileRegex(R"((?:/|^)m(\d+)_n(\d+)_tc)");
    std::smatch match;
    if (!std::regex_search(path, match, kLogTileRegex)) {
        return false;
    }
    tile_m = std::stoi(match[1].str());
    tile_n = std::stoi(match[2].str());
    return true;
}

bool parse_tile_from_combo(const std::string &combo, int &tile_m, int &tile_n) {
    static const std::regex kComboTileRegex(R"(^\s*(\d+)\s*x\s*(\d+)\s*_)");
    std::smatch match;
    if (!std::regex_search(combo, match, kComboTileRegex)) {
        return false;
    }
    tile_m = std::stoi(match[1].str());
    tile_n = std::stoi(match[2].str());
    return true;
}

std::vector<BatchSpec> load_batch_specs(const std::string &csv_path, const std::string &matrix_dir) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open batch spec file: " + csv_path);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("batch spec file is empty: " + csv_path);
    }
    const auto header = split_csv_row(line);
    size_t matrix_name_idx = header.size();
    size_t combo_idx = header.size();
    size_t log_path_idx = header.size();
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == "Matrix Name") {
            matrix_name_idx = i;
        } else if (header[i] == "FlexSpGEMM Combo") {
            combo_idx = i;
        } else if (header[i] == "FlexSpGEMM Log Path") {
            log_path_idx = i;
        }
    }
    if (matrix_name_idx == header.size()) {
        throw std::runtime_error("batch spec missing required column: Matrix Name");
    }
    if (combo_idx == header.size() && log_path_idx == header.size()) {
        throw std::runtime_error("batch spec must include FlexSpGEMM Combo and/or FlexSpGEMM Log Path");
    }

    std::vector<BatchSpec> specs;
    size_t row_idx = 1;
    while (std::getline(file, line)) {
        const auto columns = split_csv_row(line);
        if (columns.empty() || columns[0].empty()) {
            ++row_idx;
            continue;
        }
        if (matrix_name_idx >= columns.size()) {
            throw std::runtime_error("missing Matrix Name at batch spec row " + std::to_string(row_idx));
        }
        const std::string matrix_name = columns[matrix_name_idx];
        if (matrix_name.empty()) {
            ++row_idx;
            continue;
        }

        int tile_m = 0;
        int tile_n = 0;
        bool parsed = false;
        if (log_path_idx < columns.size()) {
            parsed = parse_tile_from_log_path(columns[log_path_idx], tile_m, tile_n);
        }
        if (!parsed && combo_idx < columns.size()) {
            parsed = parse_tile_from_combo(columns[combo_idx], tile_m, tile_n);
        }
        if (!parsed) {
            throw std::runtime_error("cannot parse tile sizes for matrix " + matrix_name +
                                     " at batch spec row " + std::to_string(row_idx));
        }
        std::filesystem::path mtx_path(matrix_dir);
        mtx_path /= (matrix_name + ".mtx");
        specs.push_back({matrix_name, mtx_path.string(), tile_m, tile_n});
        ++row_idx;
    }

    if (specs.empty()) {
        throw std::runtime_error("batch spec did not produce any valid entries: " + csv_path);
    }
    return specs;
}

std::string cuda_error_string(cudaError_t err, const char *expr, const char *file, int line) {
    std::ostringstream os;
    os << file << ":" << line << ": CUDA call failed: " << expr << ": "
       << cudaGetErrorString(err);
    return os.str();
}

#define CUDA_CHECK(expr)                                                                  \
    do {                                                                                  \
        cudaError_t _err = (expr);                                                        \
        if (_err != cudaSuccess) {                                                        \
            throw std::runtime_error(cuda_error_string(_err, #expr, __FILE__, __LINE__)); \
        }                                                                                 \
    } while (0)

void fill_smatrix_a(const bench::CsrMatrix &src, SMatrixA *dst) {
    std::memset(dst, 0, sizeof(SMatrixA));
    dst->m = src.rows;
    dst->n = src.cols;
    dst->nnz = static_cast<int>(src.nnz);
    dst->isSymmetric = src.symmetric ? 1 : 0;
    dst->rowpointer = static_cast<MAT_PTR_TYPE *>(std::malloc((src.rows + 1) * sizeof(MAT_PTR_TYPE)));
    dst->columnindex = static_cast<int *>(std::malloc(src.colidx.size() * sizeof(int)));
    dst->value = static_cast<MAT_VAL_TYPE *>(std::malloc(src.colidx.size() * sizeof(MAT_VAL_TYPE)));
    if (!dst->rowpointer || !dst->columnindex || !dst->value) {
        throw std::runtime_error("failed to allocate CPU CSR arrays");
    }
    for (int i = 0; i <= src.rows; ++i) {
        dst->rowpointer[i] = static_cast<MAT_PTR_TYPE>(src.rowptr[static_cast<size_t>(i)]);
    }
    for (size_t i = 0; i < src.colidx.size(); ++i) {
        dst->columnindex[i] = src.colidx[i];
        dst->value[i] = static_cast<MAT_VAL_TYPE>(i % 10);
    }
}

void fill_smatrix_b_from_a(const SMatrixA *src, SMatrixB *dst, bool aat, bool *owns_csr) {
    std::memset(dst, 0, sizeof(SMatrixB));
    *owns_csr = aat;

    if (aat) {
        dst->m = src->n;
        dst->n = src->m;
        dst->nnz = src->nnz;
        dst->rowpointer = static_cast<MAT_PTR_TYPE *>(std::malloc((dst->m + 1) * sizeof(MAT_PTR_TYPE)));
        dst->columnindex = static_cast<int *>(std::malloc(dst->nnz * sizeof(int)));
        dst->value = static_cast<MAT_VAL_TYPE *>(std::malloc(dst->nnz * sizeof(MAT_VAL_TYPE)));
        if (!dst->rowpointer || !dst->columnindex || !dst->value) {
            throw std::runtime_error("failed to allocate transposed B CSR arrays");
        }
        matrix_transposition(src->m, src->n, src->nnz,
                             src->rowpointer, src->columnindex, src->value,
                             dst->columnindex, dst->rowpointer, dst->value);
    } else {
        dst->m = src->m;
        dst->n = src->n;
        dst->nnz = src->nnz;
        dst->rowpointer = src->rowpointer;
        dst->columnindex = src->columnindex;
        dst->value = src->value;
    }
}

void free_smatrix_a(SMatrixA *matrix) {
    if (matrix->tile_ptr || matrix->device_tile_ready) {
        matrix_destroy(matrix);
    }
    std::free(matrix->rowpointer);
    std::free(matrix->columnindex);
    std::free(matrix->value);
    std::memset(matrix, 0, sizeof(SMatrixA));
}

void free_smatrix_b(SMatrixB *matrix, bool owns_csr) {
    if (matrix->tile_ptr || matrix->device_tile_ready) {
        matrix_destroy_B(matrix);
    }
    if (owns_csr) {
        std::free(matrix->rowpointer);
        std::free(matrix->columnindex);
        std::free(matrix->value);
    }
    std::memset(matrix, 0, sizeof(SMatrixB));
}

template <typename HostT, typename DeviceT>
void copy_device_to_host(HostT **host, const DeviceT *device, size_t count, const char *name) {
    if (count == 0 || device == nullptr) {
        *host = nullptr;
        return;
    }
    *host = static_cast<HostT *>(std::malloc(count * sizeof(HostT)));
    if (!*host) {
        throw std::runtime_error(std::string("failed to allocate materialized ") + name);
    }
    CUDA_CHECK(cudaMemcpy(*host, device, count * sizeof(HostT), cudaMemcpyDeviceToHost));
}

void materialize_device_tiles_a(SMatrixA *matrix, int tile_m, int tile_n) {
    if (!matrix->device_tile_ready) {
        return;
    }

    copy_device_to_host(&matrix->tile_ptr, matrix->d_tile_ptr, static_cast<size_t>(matrix->tilem + 1), "A tile_ptr");
    copy_device_to_host(&matrix->tile_columnidx, matrix->d_tile_columnidx, static_cast<size_t>(matrix->numtile), "A tile_columnidx");

    if (matrix->tile_rowidx == nullptr && matrix->numtile > 0) {
        matrix->tile_rowidx = static_cast<int *>(std::malloc(matrix->numtile * sizeof(int)));
        if (!matrix->tile_rowidx) {
            throw std::runtime_error("failed to allocate materialized A tile_rowidx");
        }
        for (int row = 0; row < matrix->tilem; ++row) {
            for (int p = matrix->tile_ptr[row]; p < matrix->tile_ptr[row + 1]; ++p) {
                matrix->tile_rowidx[p] = row;
            }
        }
    }

    copy_device_to_host(&matrix->tile_nnz, matrix->d_tile_nnz, static_cast<size_t>(matrix->numtile + 1), "A tile_nnz");
    copy_device_to_host(&matrix->tile_csr_Ptr, matrix->d_tile_csr_Ptr,
                        static_cast<size_t>(matrix->numtile) * tile_m, "A tile_csr_Ptr");
    copy_device_to_host(&matrix->tile_csr_Col, matrix->d_tile_csr_Col,
                        static_cast<size_t>(matrix->nnz), "A tile_csr_Col");
    copy_device_to_host(&matrix->tile_csr_Value, matrix->d_tile_csr_Value,
                        static_cast<size_t>(matrix->nnz), "A tile_csr_Value");
    copy_device_to_host(&matrix->mask, matrix->d_mask,
                        static_cast<size_t>(matrix->numtile) * tile_m * (tile_n / MaskBitsA), "A mask");
    copy_device_to_host(&matrix->tile_dense_ready, matrix->d_tile_dense_ready,
                        static_cast<size_t>(matrix->numtile), "A tile_dense_ready");
    copy_device_to_host(&matrix->dense_data, matrix->d_dense_data,
                        static_cast<size_t>(matrix->dense_tile_count) * tile_m * tile_n, "A dense_data");
}

void materialize_device_tiles_b(SMatrixB *matrix, int tile_m, int tile_n) {
    if (!matrix->device_tile_ready) {
        return;
    }

    copy_device_to_host(&matrix->tile_ptr, matrix->d_tile_ptr, static_cast<size_t>(matrix->tilem + 1), "B tile_ptr");
    copy_device_to_host(&matrix->tile_columnidx, matrix->d_tile_columnidx, static_cast<size_t>(matrix->numtile), "B tile_columnidx");

    if (matrix->tile_rowidx == nullptr && matrix->numtile > 0) {
        matrix->tile_rowidx = static_cast<int *>(std::malloc(matrix->numtile * sizeof(int)));
        if (!matrix->tile_rowidx) {
            throw std::runtime_error("failed to allocate materialized B tile_rowidx");
        }
        for (int row = 0; row < matrix->tilem; ++row) {
            for (int p = matrix->tile_ptr[row]; p < matrix->tile_ptr[row + 1]; ++p) {
                matrix->tile_rowidx[p] = row;
            }
        }
    }

    copy_device_to_host(&matrix->csc_tile_ptr, matrix->d_csc_tile_ptr,
                        static_cast<size_t>(matrix->tilen + 1), "B csc_tile_ptr");
    copy_device_to_host(&matrix->csc_tile_rowidx, matrix->d_csc_tile_rowidx,
                        static_cast<size_t>(matrix->numtile), "B csc_tile_rowidx");
    copy_device_to_host(&matrix->tile_nnz, matrix->d_tile_nnz, static_cast<size_t>(matrix->numtile + 1), "B tile_nnz");
    copy_device_to_host(&matrix->tile_csr_Ptr, matrix->d_tile_csr_Ptr,
                        static_cast<size_t>(matrix->numtile) * tile_n, "B tile_csr_Ptr");
    copy_device_to_host(&matrix->tile_csr_Col, matrix->d_tile_csr_Col,
                        static_cast<size_t>(matrix->nnz), "B tile_csr_Col");
    copy_device_to_host(&matrix->tile_csr_Value, matrix->d_tile_csr_Value,
                        static_cast<size_t>(matrix->nnz), "B tile_csr_Value");
    copy_device_to_host(&matrix->mask, matrix->d_mask,
                        static_cast<size_t>(matrix->numtile) * tile_n * (tile_m / MaskBitsB), "B mask");
    copy_device_to_host(&matrix->tile_dense_ready, matrix->d_tile_dense_ready,
                        static_cast<size_t>(matrix->numtile), "B tile_dense_ready");
    copy_device_to_host(&matrix->dense_data, matrix->d_dense_data,
                        static_cast<size_t>(matrix->dense_tile_count) * tile_n * tile_m, "B dense_data");
}

template <typename CpuT, typename GpuT>
int count_mismatches(const CpuT *cpu, const GpuT *gpu, int n, int max_examples, const char *name) {
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (static_cast<GpuT>(cpu[i]) != gpu[i]) {
            if (mismatches < max_examples) {
                std::cout << "    " << name << " mismatch at " << i
                          << ": cpu=" << +cpu[i]
                          << ", gpu=" << +gpu[i] << "\n";
            }
            ++mismatches;
        }
    }
    return mismatches;
}

int compare_a(const SMatrixA &cpu, const SMatrixA &gpu, int tile_m, int tile_n, const char *prefix) {
    int mismatches = 0;
    if (cpu.tilem != gpu.tilem || cpu.tilen != gpu.tilen || cpu.numtile != gpu.numtile ||
        cpu.dense_tile_count != gpu.dense_tile_count) {
        std::cout << "    " << prefix << "_meta mismatch"
                  << ": cpu(tilem=" << cpu.tilem << ", tilen=" << cpu.tilen
                  << ", tiles=" << cpu.numtile << ", dense=" << cpu.dense_tile_count
                  << "), gpu(tilem=" << gpu.tilem << ", tilen=" << gpu.tilen
                  << ", tiles=" << gpu.numtile << ", dense=" << gpu.dense_tile_count << ")\n";
        ++mismatches;
    }

    if (cpu.tilem == gpu.tilem) {
        mismatches += count_mismatches(cpu.tile_ptr, gpu.tile_ptr, cpu.tilem + 1, 5, (std::string(prefix) + "_tile_ptr").c_str());
    }
    if (cpu.numtile == gpu.numtile) {
        mismatches += count_mismatches(cpu.tile_columnidx, gpu.tile_columnidx, cpu.numtile, 5, (std::string(prefix) + "_tile_columnidx").c_str());
        mismatches += count_mismatches(cpu.tile_rowidx, gpu.tile_rowidx, cpu.numtile, 5, (std::string(prefix) + "_tile_rowidx").c_str());
        mismatches += count_mismatches(cpu.tile_nnz, gpu.tile_nnz, cpu.numtile + 1, 5, (std::string(prefix) + "_tile_nnz").c_str());
        mismatches += count_mismatches(cpu.tile_csr_Ptr, gpu.tile_csr_Ptr, cpu.numtile * tile_m, 5, (std::string(prefix) + "_tile_csr_Ptr").c_str());
        mismatches += count_mismatches(cpu.mask, gpu.mask, cpu.numtile * tile_m * (tile_n / MaskBitsA), 5, (std::string(prefix) + "_mask").c_str());
        mismatches += count_mismatches(cpu.tile_dense_ready, gpu.tile_dense_ready, cpu.numtile, 5, (std::string(prefix) + "_tile_dense_ready").c_str());
    }
    mismatches += count_mismatches(cpu.tile_csr_Col, gpu.tile_csr_Col, cpu.nnz, 5, (std::string(prefix) + "_tile_csr_Col").c_str());
    mismatches += count_mismatches(cpu.tile_csr_Value, gpu.tile_csr_Value, cpu.nnz, 5, (std::string(prefix) + "_tile_csr_Value").c_str());
    if (cpu.dense_tile_count == gpu.dense_tile_count) {
        mismatches += count_mismatches(cpu.dense_data, gpu.dense_data,
                                       cpu.dense_tile_count * tile_m * tile_n, 5,
                                       (std::string(prefix) + "_dense_data").c_str());
    }
    return mismatches;
}

int compare_b(const SMatrixB &cpu, const SMatrixB &gpu, int tile_m, int tile_n, const char *prefix) {
    int mismatches = 0;
    if (cpu.tilem != gpu.tilem || cpu.tilen != gpu.tilen || cpu.numtile != gpu.numtile ||
        cpu.dense_tile_count != gpu.dense_tile_count) {
        std::cout << "    " << prefix << "_meta mismatch"
                  << ": cpu(tilem=" << cpu.tilem << ", tilen=" << cpu.tilen
                  << ", tiles=" << cpu.numtile << ", dense=" << cpu.dense_tile_count
                  << "), gpu(tilem=" << gpu.tilem << ", tilen=" << gpu.tilen
                  << ", tiles=" << gpu.numtile << ", dense=" << gpu.dense_tile_count << ")\n";
        ++mismatches;
    }

    if (cpu.tilem == gpu.tilem) {
        mismatches += count_mismatches(cpu.tile_ptr, gpu.tile_ptr, cpu.tilem + 1, 5, (std::string(prefix) + "_tile_ptr").c_str());
    }
    if (cpu.tilen == gpu.tilen) {
        mismatches += count_mismatches(cpu.csc_tile_ptr, gpu.csc_tile_ptr, cpu.tilen + 1, 5, (std::string(prefix) + "_csc_tile_ptr").c_str());
    }
    if (cpu.numtile == gpu.numtile) {
        mismatches += count_mismatches(cpu.tile_columnidx, gpu.tile_columnidx, cpu.numtile, 5, (std::string(prefix) + "_tile_columnidx").c_str());
        mismatches += count_mismatches(cpu.csc_tile_rowidx, gpu.csc_tile_rowidx, cpu.numtile, 5, (std::string(prefix) + "_csc_tile_rowidx").c_str());
        mismatches += count_mismatches(cpu.tile_nnz, gpu.tile_nnz, cpu.numtile + 1, 5, (std::string(prefix) + "_tile_nnz").c_str());
        mismatches += count_mismatches(cpu.tile_csr_Ptr, gpu.tile_csr_Ptr, cpu.numtile * tile_n, 5, (std::string(prefix) + "_tile_csr_Ptr").c_str());
        mismatches += count_mismatches(cpu.mask, gpu.mask, cpu.numtile * tile_n * (tile_m / MaskBitsB), 5, (std::string(prefix) + "_mask").c_str());
        mismatches += count_mismatches(cpu.tile_dense_ready, gpu.tile_dense_ready, cpu.numtile, 5, (std::string(prefix) + "_tile_dense_ready").c_str());
    }
    mismatches += count_mismatches(cpu.tile_csr_Col, gpu.tile_csr_Col, cpu.nnz, 5, (std::string(prefix) + "_tile_csr_Col").c_str());
    mismatches += count_mismatches(cpu.tile_csr_Value, gpu.tile_csr_Value, cpu.nnz, 5, (std::string(prefix) + "_tile_csr_Value").c_str());
    if (cpu.dense_tile_count == gpu.dense_tile_count) {
        mismatches += count_mismatches(cpu.dense_data, gpu.dense_data,
                                       cpu.dense_tile_count * tile_n * tile_m, 5,
                                       (std::string(prefix) + "_dense_data").c_str());
    }
    return mismatches;
}

unsigned long long compute_nnz_upper_bound(const SMatrixA &a, const SMatrixB &b) {
    unsigned long long nnzCub = 0;
    for (int i = 0; i < a.nnz; ++i) {
        const int rowidx = a.columnindex[i];
        nnzCub += static_cast<unsigned long long>(b.rowpointer[rowidx + 1] - b.rowpointer[rowidx]);
    }
    return nnzCub;
}

int run_matrix(const std::string &path,
               int tile_m,
               int tile_n,
               int aat,
               double tau,
               bool timing_only,
               bool gpu_only,
               bool device_output,
               int repeat,
               bool profile_stages) {
    bench::Timer load_timer;
    bench::CsrMatrix matrix = bench::load_matrix_market(path);
    const double load_ms = load_timer.elapsed_ms();

    if (!aat && matrix.rows != matrix.cols) {
        throw std::runtime_error("matrix squaring requires rowA == colA");
    }
    if (aat && matrix.rows == matrix.cols && matrix.symmetric) {
        throw std::runtime_error("AAT does not support symmetric matrix, matching main.cu");
    }

    SMatrixA cpu_a;
    SMatrixA gpu_a;
    SMatrixB cpu_b;
    SMatrixB gpu_b;
    bool cpu_a_ready = false;
    bool gpu_a_ready = false;
    bool cpu_b_ready = false;
    bool gpu_b_ready = false;
    bool cpu_b_owns_csr = false;
    bool gpu_b_owns_csr = false;
    if (repeat <= 0) {
        repeat = 1;
    }

    double cpu_a_ms = 0.0;
    double gpu_a_ms = 0.0;
    double gpu_b_ms = 0.0;
    double gpu_a_min_ms = 0.0;
    double gpu_a_max_ms = 0.0;
    double gpu_a_avg_ms = 0.0;
    double gpu_b_min_ms = 0.0;
    double gpu_b_max_ms = 0.0;
    double gpu_b_avg_ms = 0.0;
    double h2d_a_ms = 0.0;
    double h2d_b_ms = 0.0;
    unsigned long long nnzCub = 0;
    int failures = 0;

    auto median_ms = [](std::vector<double> values) {
        if (values.empty()) {
            return 0.0;
        }
        std::sort(values.begin(), values.end());
        const size_t mid = values.size() / 2;
        if ((values.size() & 1u) == 1u) {
            return values[mid];
        }
        return 0.5 * (values[mid - 1] + values[mid]);
    };

    auto summarize_ms = [&](const std::vector<double> &vals, double &min_ms, double &max_ms, double &avg_ms) {
        if (vals.empty()) {
            min_ms = 0.0;
            max_ms = 0.0;
            avg_ms = 0.0;
            return 0.0;
        }
        auto minmax = std::minmax_element(vals.begin(), vals.end());
        min_ms = *minmax.first;
        max_ms = *minmax.second;
        double sum = 0.0;
        for (double v : vals) {
            sum += v;
        }
        avg_ms = sum / static_cast<double>(vals.size());
        return median_ms(vals);
    };

    using StageTimes = gpu_csr2tile::Csr2TileStageTimes;
    auto median_stage_field = [&](const std::vector<StageTimes> &vals, double StageTimes::*field) {
        std::vector<double> samples;
        samples.reserve(vals.size());
        for (const auto &v : vals) {
            samples.push_back(v.*field);
        }
        return median_ms(samples);
    };
    auto summarize_stage = [&](const std::vector<StageTimes> &vals) {
        StageTimes out;
        if (vals.empty()) {
            return out;
        }
        out.h2d_ms = median_stage_field(vals, &StageTimes::h2d_ms);
        out.row_structure_ms = median_stage_field(vals, &StageTimes::row_structure_ms);
        out.fill_entry_keys_ms = median_stage_field(vals, &StageTimes::fill_entry_keys_ms);
        out.sort_by_key_ms = median_stage_field(vals, &StageTimes::sort_by_key_ms);
        out.reduce_by_key_ms = median_stage_field(vals, &StageTimes::reduce_by_key_ms);
        out.tile_index_ms = median_stage_field(vals, &StageTimes::tile_index_ms);
        out.tile_csr_ptr_ms = median_stage_field(vals, &StageTimes::tile_csr_ptr_ms);
        out.write_entries_ms = median_stage_field(vals, &StageTimes::write_entries_ms);
        out.dense_flags_ms = median_stage_field(vals, &StageTimes::dense_flags_ms);
        out.mask_dense_ms = median_stage_field(vals, &StageTimes::mask_dense_ms);
        out.narrow_mask_ms = median_stage_field(vals, &StageTimes::narrow_mask_ms);
        out.d2d_output_copy_ms = median_stage_field(vals, &StageTimes::d2d_output_copy_ms);
        out.build_total_ms = median_stage_field(vals, &StageTimes::build_total_ms);
        return out;
    };

    auto build_gpu_inputs = [&]() {
        if (gpu_a_ready) {
            free_smatrix_a(&gpu_a);
            gpu_a_ready = false;
        }
        if (gpu_b_ready) {
            free_smatrix_b(&gpu_b, gpu_b_owns_csr);
            gpu_b_ready = false;
        }
        fill_smatrix_a(matrix, &gpu_a);
        gpu_a_ready = true;
        fill_smatrix_b_from_a(&gpu_a, &gpu_b, aat != 0, &gpu_b_owns_csr);
        gpu_b_ready = true;
    };

    try {
        if (!gpu_only) {
            fill_smatrix_a(matrix, &cpu_a);
            cpu_a_ready = true;
            fill_smatrix_b_from_a(&cpu_a, &cpu_b, aat != 0, &cpu_b_owns_csr);
            cpu_b_ready = true;
            nnzCub = compute_nnz_upper_bound(cpu_a, cpu_b);

            bench::Timer cpu_a_timer;
            csr2tile_row_major(&cpu_a, tile_m, tile_n);
            cpu_a_ms = cpu_a_timer.elapsed_ms();

            if (!timing_only) {
                csr2tile_col_major(&cpu_b, tile_m, tile_n);
            }
        }

        build_gpu_inputs();
        if (gpu_only) {
            nnzCub = compute_nnz_upper_bound(gpu_a, gpu_b);
        }

        std::vector<double> gpu_a_samples;
        std::vector<double> gpu_b_samples;
        std::vector<StageTimes> gpu_a_stage_samples;
        std::vector<StageTimes> gpu_b_stage_samples;
        gpu_a_samples.reserve(static_cast<size_t>(repeat));
        gpu_b_samples.reserve(static_cast<size_t>(repeat));
        gpu_a_stage_samples.reserve(static_cast<size_t>(repeat));
        gpu_b_stage_samples.reserve(static_cast<size_t>(repeat));
        for (int repeat_i = 0; repeat_i < repeat; ++repeat_i) {
            if (repeat_i > 0) {
                build_gpu_inputs();
            }

            StageTimes gpu_a_stage;
            bench::Timer gpu_a_timer;
            if (device_output) {
                gpu_csr2tile::gpu_csr2tile_row_major_device(
                    &gpu_a, tile_m, tile_n, &h2d_a_ms, profile_stages ? &gpu_a_stage : nullptr);
            } else {
                gpu_csr2tile::gpu_csr2tile_row_major(&gpu_a, tile_m, tile_n, &h2d_a_ms);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            gpu_a_ms = gpu_a_timer.elapsed_ms() - h2d_a_ms;
            if (gpu_a_ms < 0.0) {
                gpu_a_ms = 0.0;
            }
            gpu_a_samples.push_back(gpu_a_ms);
            if (profile_stages && device_output) {
                gpu_a_stage_samples.push_back(gpu_a_stage);
            }

            StageTimes gpu_b_stage;
            bench::Timer gpu_b_timer;
            if (device_output) {
                gpu_csr2tile::gpu_csr2tile_col_major_device(
                    &gpu_b, tile_m, tile_n, &h2d_b_ms, profile_stages ? &gpu_b_stage : nullptr);
            } else {
                gpu_csr2tile::gpu_csr2tile_col_major(&gpu_b, tile_m, tile_n, &h2d_b_ms);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            gpu_b_ms = gpu_b_timer.elapsed_ms() - h2d_b_ms;
            if (gpu_b_ms < 0.0) {
                gpu_b_ms = 0.0;
            }
            gpu_b_samples.push_back(gpu_b_ms);
            if (profile_stages && device_output) {
                gpu_b_stage_samples.push_back(gpu_b_stage);
            }
        }
        gpu_a_ms = summarize_ms(gpu_a_samples, gpu_a_min_ms, gpu_a_max_ms, gpu_a_avg_ms);
        gpu_b_ms = summarize_ms(gpu_b_samples, gpu_b_min_ms, gpu_b_max_ms, gpu_b_avg_ms);
        const StageTimes gpu_a_stage = summarize_stage(gpu_a_stage_samples);
        const StageTimes gpu_b_stage = summarize_stage(gpu_b_stage_samples);

        if (timing_only) {
            std::cout << bench::basename(path)
                      << ": TIMING"
                      << ", rows=" << matrix.rows
                      << ", cols=" << matrix.cols
                      << ", nnz=" << matrix.nnz
                      << ", aat=" << aat
                      << ", tau=" << tau
                      << ", tc_threshold=" << (tau == 0.0 ? 1 : static_cast<int>(tau * tile_m * tile_m))
                      << ", nnzCub=" << nnzCub
                      << ", load_ms=" << load_ms
                      << ", cpu_ms=";
            if (!gpu_only) {
                std::cout << cpu_a_ms;
            }
            std::cout << ", gpu_ms=" << gpu_a_ms
                      << ", gpu_b_ms=" << gpu_b_ms
                      << ", gpu_ms_min=" << gpu_a_min_ms
                      << ", gpu_ms_max=" << gpu_a_max_ms
                      << ", gpu_ms_avg=" << gpu_a_avg_ms
                      << ", gpu_b_ms_median=" << gpu_b_ms
                      << ", gpu_b_ms_min=" << gpu_b_min_ms
                      << ", gpu_b_ms_max=" << gpu_b_max_ms
                      << ", gpu_b_ms_avg=" << gpu_b_avg_ms
                      << ", repeat=" << repeat
                      << ", profile_stages=" << (profile_stages && device_output ? 1 : 0);
            if (profile_stages && device_output) {
                std::cout << ", gpu_h2d_ms_median=" << gpu_a_stage.h2d_ms
                          << ", gpu_stage_build_total_ms=" << gpu_a_stage.build_total_ms
                          << ", gpu_stage_fill_entry_keys_ms=" << gpu_a_stage.fill_entry_keys_ms
                          << ", gpu_stage_sort_by_key_ms=" << gpu_a_stage.sort_by_key_ms
                          << ", gpu_stage_reduce_by_key_ms=" << gpu_a_stage.reduce_by_key_ms
                          << ", gpu_stage_tile_index_ms=" << gpu_a_stage.tile_index_ms
                          << ", gpu_stage_tile_csr_ptr_ms=" << gpu_a_stage.tile_csr_ptr_ms
                          << ", gpu_stage_write_entries_ms=" << gpu_a_stage.write_entries_ms
                          << ", gpu_stage_dense_flags_ms=" << gpu_a_stage.dense_flags_ms
                          << ", gpu_stage_mask_dense_ms=" << gpu_a_stage.mask_dense_ms
                          << ", gpu_stage_narrow_mask_ms=" << gpu_a_stage.narrow_mask_ms
                          << ", gpu_stage_d2d_output_copy_ms=" << gpu_a_stage.d2d_output_copy_ms
                          << ", gpu_b_h2d_ms_median=" << gpu_b_stage.h2d_ms
                          << ", gpu_b_stage_row_structure_ms=" << gpu_b_stage.row_structure_ms
                          << ", gpu_b_stage_build_total_ms=" << gpu_b_stage.build_total_ms
                          << ", gpu_b_stage_fill_entry_keys_ms=" << gpu_b_stage.fill_entry_keys_ms
                          << ", gpu_b_stage_sort_by_key_ms=" << gpu_b_stage.sort_by_key_ms
                          << ", gpu_b_stage_reduce_by_key_ms=" << gpu_b_stage.reduce_by_key_ms
                          << ", gpu_b_stage_tile_index_ms=" << gpu_b_stage.tile_index_ms
                          << ", gpu_b_stage_tile_csr_ptr_ms=" << gpu_b_stage.tile_csr_ptr_ms
                          << ", gpu_b_stage_write_entries_ms=" << gpu_b_stage.write_entries_ms
                          << ", gpu_b_stage_dense_flags_ms=" << gpu_b_stage.dense_flags_ms
                          << ", gpu_b_stage_mask_dense_ms=" << gpu_b_stage.mask_dense_ms
                          << ", gpu_b_stage_narrow_mask_ms=" << gpu_b_stage.narrow_mask_ms
                          << ", gpu_b_stage_d2d_output_copy_ms=" << gpu_b_stage.d2d_output_copy_ms;
            }
            std::cout << ", cpu_over_gpu=";
            if (!gpu_only) {
                std::cout << (gpu_a_ms > 0.0 ? cpu_a_ms / gpu_a_ms : 0.0);
            }
            std::cout << ", gpu_over_cpu=";
            if (!gpu_only) {
                std::cout << (cpu_a_ms > 0.0 ? gpu_a_ms / cpu_a_ms : 0.0);
            }
            std::cout << ", cpu_tiles=";
            if (!gpu_only) {
                std::cout << cpu_a.numtile;
            }
            std::cout << ", gpu_tiles=" << gpu_a.numtile
                      << ", gpu_b_tiles=" << gpu_b.numtile
                      << ", cpu_dense_tiles=";
            if (!gpu_only) {
                std::cout << cpu_a.dense_tile_count;
            }
            std::cout << ", gpu_dense_tiles=" << gpu_a.dense_tile_count
                      << ", gpu_b_dense_tiles=" << gpu_b.dense_tile_count << "\n";
        } else {
            if (gpu_only) {
                throw std::runtime_error("--gpu-only cannot be used with --check");
            }
            if (device_output) {
                materialize_device_tiles_a(&gpu_a, tile_m, tile_n);
                materialize_device_tiles_b(&gpu_b, tile_m, tile_n);
            }

            const int a_mismatches = compare_a(cpu_a, gpu_a, tile_m, tile_n, "a");
            const int b_mismatches = compare_b(cpu_b, gpu_b, tile_m, tile_n, "b");
            const bool ok = (a_mismatches == 0 && b_mismatches == 0);
            failures = ok ? 0 : 1;

            std::cout << bench::basename(path)
                      << ": " << (ok ? "PASS" : "FAIL")
                      << ", rows=" << matrix.rows
                      << ", cols=" << matrix.cols
                      << ", nnz=" << matrix.nnz
                      << ", aat=" << aat
                      << ", tau=" << tau
                      << ", tc_threshold=" << (tau == 0.0 ? 1 : static_cast<int>(tau * tile_m * tile_m))
                      << ", nnzCub=" << nnzCub
                      << ", load_ms=" << load_ms
                      << ", cpu_ms=" << cpu_a_ms
                      << ", gpu_ms=" << gpu_a_ms
                      << ", gpu_b_ms=" << gpu_b_ms
                      << ", cpu_tiles=" << cpu_a.numtile
                      << ", gpu_tiles=" << gpu_a.numtile
                      << ", gpu_b_tiles=" << gpu_b.numtile
                      << ", cpu_dense_tiles=" << cpu_a.dense_tile_count
                      << ", gpu_dense_tiles=" << gpu_a.dense_tile_count
                      << ", gpu_b_dense_tiles=" << gpu_b.dense_tile_count
                      << ", a_mismatches=" << a_mismatches
                      << ", b_mismatches=" << b_mismatches
                      << "\n";
        }
    } catch (...) {
        if (cpu_b_ready) {
            free_smatrix_b(&cpu_b, cpu_b_owns_csr);
        }
        if (gpu_b_ready) {
            free_smatrix_b(&gpu_b, gpu_b_owns_csr);
        }
        if (cpu_a_ready) {
            free_smatrix_a(&cpu_a);
        }
        if (gpu_a_ready) {
            free_smatrix_a(&gpu_a);
        }
        throw;
    }

    if (cpu_b_ready) {
        free_smatrix_b(&cpu_b, cpu_b_owns_csr);
    }
    if (gpu_b_ready) {
        free_smatrix_b(&gpu_b, gpu_b_owns_csr);
    }
    if (cpu_a_ready) {
        free_smatrix_a(&cpu_a);
    }
    if (gpu_a_ready) {
        free_smatrix_a(&gpu_a);
    }
    return failures;
}

void usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " [-d|--device N] [-aat <0|1>] [-tau R] [--tile-m M] [--tile-n N]\n"
              << "       [--gpu-only] [--check] [--timing-only] [--repeat N] [--profile-stages]\n"
              << "       [--host-output|--device-output] [--batch-spec <fig8-csv>] [--matrix-dir <path>]\n"
              << "       <matrix.mtx>...\n";
}

}  // namespace

int main(int argc, char **argv) {
    int device = 0;
    int tile_m = TILE_SIZE_M;
    int tile_n = TILE_SIZE_N;
    int aat = 0;
    double tau = 0.7;
    bool timing_only = true;
    bool gpu_only = false;
    bool device_output = true;
    bool profile_stages = false;
    int repeat = 5;
    std::string batch_spec_path;
    std::string matrix_dir = ".";
    std::vector<std::string> paths;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" || arg == "-d") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            device = std::atoi(argv[i]);
        } else if (arg == "-aat" || arg == "--aat") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            aat = std::atoi(argv[i]);
        } else if (arg == "-tau" || arg == "--tau") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            tau = std::atof(argv[i]);
        } else if (arg == "--tile-m") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            tile_m = std::atoi(argv[i]);
        } else if (arg == "--tile-n") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            tile_n = std::atoi(argv[i]);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--check") {
            timing_only = false;
        } else if (arg == "--timing-only") {
            timing_only = true;
        } else if (arg == "--gpu-only") {
            gpu_only = true;
        } else if (arg == "--profile-stages") {
            profile_stages = true;
        } else if (arg == "--repeat") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            repeat = std::atoi(argv[i]);
        } else if (arg == "--device-output") {
            device_output = true;
        } else if (arg == "--host-output") {
            device_output = false;
        } else if (arg == "--batch-spec") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            batch_spec_path = argv[i];
        } else if (arg == "--matrix-dir") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            matrix_dir = argv[i];
        } else {
            paths.push_back(arg);
        }
    }

    if (aat != 0 && aat != 1) {
        std::cerr << "Error: -aat must be 0 or 1\n";
        return 1;
    }
    if (repeat <= 0) {
        std::cerr << "Error: --repeat must be >= 1\n";
        return 1;
    }
    if (!batch_spec_path.empty() && !paths.empty()) {
        std::cerr << "Error: --batch-spec cannot be used with positional matrix paths\n";
        return 1;
    }
    if (batch_spec_path.empty() && paths.empty()) {
        usage(argv[0]);
        return 1;
    }

    try {
        CUDA_CHECK(cudaSetDevice(device));
        int failures = 0;
        if (!batch_spec_path.empty()) {
            const auto specs = load_batch_specs(batch_spec_path, matrix_dir);
            for (size_t idx = 0; idx < specs.size(); ++idx) {
                const auto &spec = specs[idx];
                std::cout << "[batch "
                          << (idx + 1) << "/" << specs.size()
                          << "] " << spec.matrix_name
                          << ", tile=" << spec.tile_m << "x" << spec.tile_n
                          << ", path=" << spec.matrix_path << "\n";
                try {
                    failures += run_matrix(spec.matrix_path, spec.tile_m, spec.tile_n, aat, tau,
                                          timing_only, gpu_only, device_output, repeat, profile_stages);
                } catch (const std::exception &e) {
                    ++failures;
                    std::cerr << "Error: failed matrix " << spec.matrix_name
                              << ": " << e.what() << "\n";
                }
            }
        } else {
            for (const std::string &path : paths) {
                try {
                    failures += run_matrix(path, tile_m, tile_n, aat, tau, timing_only,
                                          gpu_only, device_output, repeat, profile_stages);
                } catch (const std::exception &e) {
                    ++failures;
                    std::cerr << "Error: failed matrix " << path << ": " << e.what() << "\n";
                }
            }
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

#undef CUDA_CHECK
