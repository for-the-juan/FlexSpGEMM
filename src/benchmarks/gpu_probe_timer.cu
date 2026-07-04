#include "probe_common.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace {

using Key = unsigned long long;

const int kAProbeTileSizes[][2] = {
    {8, 8}, {16, 8}, {8, 16}, {16, 16}, {16, 32},
    {32, 16}, {32, 32}, {8, 32}, {32, 8},
};
const int kCProbeTileMs[] = {8, 16, 32};
constexpr int kCProbeHashCapacity = 4096;
constexpr int kCProbeAvgAllSegmentLimit = 10;

std::string cuda_error_string(cudaError_t err, const char *expr, const char *file, int line) {
    std::ostringstream os;
    os << file << ":" << line << ": CUDA call failed: " << expr << ": "
       << cudaGetErrorString(err);
    return os.str();
}

std::string cusparse_error_string(cusparseStatus_t st, const char *expr, const char *file, int line) {
    std::ostringstream os;
    os << file << ":" << line << ": cuSPARSE call failed: " << expr << ": "
       << static_cast<int>(st);
    return os.str();
}

#define CUDA_CHECK(expr)                                                                       \
    do {                                                                                       \
        cudaError_t _err = (expr);                                                             \
        if (_err != cudaSuccess) {                                                             \
            throw std::runtime_error(cuda_error_string(_err, #expr, __FILE__, __LINE__));      \
        }                                                                                      \
    } while (0)

#define CUSPARSE_CHECK(expr)                                                                   \
    do {                                                                                       \
        cusparseStatus_t _st = (expr);                                                         \
        if (_st != CUSPARSE_STATUS_SUCCESS) {                                                  \
            throw std::runtime_error(cusparse_error_string(_st, #expr, __FILE__, __LINE__));   \
        }                                                                                      \
    } while (0)

int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

template <typename Fn>
double timed_cuda_ms(Fn &&fn) {
    CUDA_CHECK(cudaDeviceSynchronize());
    bench::Timer timer;
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    return timer.elapsed_ms();
}

struct DeviceCsr {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    thrust::device_vector<int> rowptr;
    thrust::device_vector<int> colidx;
};

struct TileAccumValue {
    int nnz;
    unsigned int row_mask;
    unsigned int col_mask;
};

struct TileAccumPlus {
    __host__ __device__ TileAccumValue operator()(const TileAccumValue &a,
                                                  const TileAccumValue &b) const {
        TileAccumValue out;
        out.nnz = a.nnz + b.nnz;
        out.row_mask = a.row_mask | b.row_mask;
        out.col_mask = a.col_mask | b.col_mask;
        return out;
    }
};

struct UniqueKeys {
    thrust::device_vector<Key> keys;
    thrust::device_vector<int> counts;
};

struct UniqueTileAccum {
    thrust::device_vector<Key> keys;
    thrust::device_vector<TileAccumValue> values;
};

struct BaseTileAccumGpu {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    thrust::device_vector<Key> keys;
    thrust::device_vector<TileAccumValue> values;
};

struct MatrixResult {
    std::string matrix;
    int rows = 0;
    int cols = 0;
    long long nnz = 0;
    int symmetric = 0;
    double load_ms = 0.0;
    double h2d_ms = 0.0;
    double a_probe_ms = 0.0;
    long long a_tiles_16x16 = 0;
    double c_build_ms = 0.0;
    double c_feature_ms = 0.0;
    long long c_tiles_8 = 0;
    double c_avg_8 = 0.0;
    int c_max_8 = 0;
    long long c_tiles_16 = 0;
    double c_avg_16 = 0.0;
    int c_max_16 = 0;
    long long c_tiles_32 = 0;
    double c_avg_32 = 0.0;
    int c_max_32 = 0;
    int c_hash_overflow_rows = 0;
    int c_hash_fallback_rows = 0;
    double csr2tile_pattern_ms = 0.0;
    long long csr2tile_tiles = 0;
};

enum class CProbeImpl {
    Hash,
    HashAvg,
    Cusparse,
};

enum class AProbeImpl {
    Merge8,
    Packed,
    Sort3,
};

const char *a_probe_impl_name(AProbeImpl impl) {
    switch (impl) {
    case AProbeImpl::Merge8:
        return "merge8";
    case AProbeImpl::Packed:
        return "packed";
    case AProbeImpl::Sort3:
        return "sort3";
    }
    return "unknown";
}

const char *c_probe_impl_name(CProbeImpl impl) {
    switch (impl) {
        case CProbeImpl::Hash:
            return "hash";
        case CProbeImpl::HashAvg:
            return "hash-avg";
        case CProbeImpl::Cusparse:
            return "cusparse";
    }
    return "unknown";
}

struct PackedTileKey {
    __host__ __device__ Key operator()(const Key x) const {
        return x >> 10;
    }
};

struct PackedMajorSlotKey {
    __host__ __device__ Key operator()(const Key x) const {
        return ((x >> 10) << 5) | ((x >> 5) & 31ULL);
    }
};

__host__ __device__ unsigned int popcount_u32(unsigned int v) {
    unsigned int count = 0;
    while (v) {
        count += v & 1U;
        v >>= 1U;
    }
    return count;
}

struct RowSlotCount {
    __host__ __device__ unsigned long long operator()(const TileAccumValue &v) const {
        return static_cast<unsigned long long>(popcount_u32(v.row_mask));
    }
};

struct ColSlotCount {
    __host__ __device__ unsigned long long operator()(const TileAccumValue &v) const {
        return static_cast<unsigned long long>(popcount_u32(v.col_mask));
    }
};

__global__ void fill_tile_keys_kernel(int rows,
                                      const int *rowptr,
                                      const int *colidx,
                                      int tile_m,
                                      int tile_n,
                                      int tile_cols,
                                      Key *tile_keys,
                                      Key *row_slot_keys,
                                      Key *col_slot_keys) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }
    const int tile_row = row / tile_m;
    const int local_row = row - tile_row * tile_m;
    for (int p = rowptr[row]; p < rowptr[row + 1]; ++p) {
        const int col = colidx[p];
        const int tile_col = col / tile_n;
        const int local_col = col - tile_col * tile_n;
        const Key tile_key = static_cast<Key>(tile_row) * static_cast<Key>(tile_cols) +
                             static_cast<Key>(tile_col);
        tile_keys[p] = tile_key;
        if (row_slot_keys) {
            row_slot_keys[p] = tile_key * static_cast<Key>(tile_m) + static_cast<Key>(local_row);
        }
        if (col_slot_keys) {
            col_slot_keys[p] = tile_key * static_cast<Key>(tile_n) + static_cast<Key>(local_col);
        }
    }
}

__global__ void fill_base8_accum_keys_kernel(int rows,
                                             const int *rowptr,
                                             const int *colidx,
                                             int base_cols,
                                             Key *keys,
                                             TileAccumValue *values) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }
    const int tile_row = row >> 3;
    const int local_row = row & 7;
    for (int p = rowptr[row]; p < rowptr[row + 1]; ++p) {
        const int col = colidx[p];
        const int tile_col = col >> 3;
        const int local_col = col & 7;
        keys[p] = static_cast<Key>(tile_row) * static_cast<Key>(base_cols) +
                  static_cast<Key>(tile_col);
        TileAccumValue v;
        v.nnz = 1;
        v.row_mask = 1U << local_row;
        v.col_mask = 1U << local_col;
        values[p] = v;
    }
}

__global__ void merge_a_probe_accum_keys_kernel(int base_nnz,
                                                int base_cols,
                                                const Key *base_keys,
                                                const TileAccumValue *base_values,
                                                int row_factor,
                                                int col_factor,
                                                int merged_cols,
                                                Key *merged_keys,
                                                TileAccumValue *merged_values) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= base_nnz) {
        return;
    }

    const Key base_key = base_keys[idx];
    const int base_row = static_cast<int>(base_key / static_cast<Key>(base_cols));
    const int base_col = static_cast<int>(base_key % static_cast<Key>(base_cols));
    const int merged_row = base_row / row_factor;
    const int merged_col = base_col / col_factor;
    const int row_offset = (base_row % row_factor) * 8;
    const int col_offset = (base_col % col_factor) * 8;

    merged_keys[idx] = static_cast<Key>(merged_row) * static_cast<Key>(merged_cols) +
                       static_cast<Key>(merged_col);
    TileAccumValue v = base_values[idx];
    v.row_mask <<= row_offset;
    v.col_mask <<= col_offset;
    merged_values[idx] = v;
}

__global__ void fill_packed_a_probe_keys_kernel(int rows,
                                                const int *rowptr,
                                                const int *colidx,
                                                int tile_m,
                                                int tile_n,
                                                int tile_cols,
                                                Key *row_major_keys,
                                                Key *col_major_keys) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }
    const int tile_row = row / tile_m;
    const int local_row = row - tile_row * tile_m;
    for (int p = rowptr[row]; p < rowptr[row + 1]; ++p) {
        const int col = colidx[p];
        const int tile_col = col / tile_n;
        const int local_col = col - tile_col * tile_n;
        const Key tile_key = static_cast<Key>(tile_row) * static_cast<Key>(tile_cols) +
                             static_cast<Key>(tile_col);
        row_major_keys[p] = (tile_key << 10) |
                            (static_cast<Key>(local_row) << 5) |
                            static_cast<Key>(local_col);
        col_major_keys[p] = (tile_key << 10) |
                            (static_cast<Key>(local_col) << 5) |
                            static_cast<Key>(local_row);
    }
}

__global__ void fill_transposed_tile_keys_kernel(int rows,
                                                 const int *rowptr,
                                                 const int *colidx,
                                                 int tile_m,
                                                 int tile_n,
                                                 int tile_cols,
                                                 Key *tile_keys) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }
    for (int p = rowptr[row]; p < rowptr[row + 1]; ++p) {
        const int col = colidx[p];
        const int tile_row = col / tile_m;
        const int tile_col = row / tile_n;
        tile_keys[p] = static_cast<Key>(tile_row) * static_cast<Key>(tile_cols) +
                       static_cast<Key>(tile_col);
    }
}

__global__ void count_tile_rows_kernel(const Key *keys,
                                       int nkeys,
                                       int tile_cols,
                                       int *rowptr_counts) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nkeys) {
        return;
    }
    const int row = static_cast<int>(keys[idx] / static_cast<Key>(tile_cols));
    atomicAdd(rowptr_counts + row + 1, 1);
}

__global__ void keys_to_cols_kernel(const Key *keys, int nkeys, int tile_cols, int *cols) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nkeys) {
        return;
    }
    cols[idx] = static_cast<int>(keys[idx] % static_cast<Key>(tile_cols));
}

__global__ void merge_tile_keys_kernel(int base_rows,
                                       const int *base_rowptr,
                                       const int *base_colidx,
                                       int row_factor,
                                       int col_factor,
                                       int merged_cols,
                                       Key *merged_keys) {
    const int base_row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (base_row >= base_rows) {
        return;
    }
    const int merged_row = base_row / row_factor;
    for (int p = base_rowptr[base_row]; p < base_rowptr[base_row + 1]; ++p) {
        const int merged_col = base_colidx[p] / col_factor;
        merged_keys[p] = static_cast<Key>(merged_row) * static_cast<Key>(merged_cols) +
                         static_cast<Key>(merged_col);
    }
}

UniqueKeys reduce_unique_keys(thrust::device_vector<Key> keys) {
    thrust::sort(thrust::device, keys.begin(), keys.end());

    UniqueKeys unique;
    unique.keys.resize(keys.size());
    unique.counts.resize(keys.size());
    auto ends = thrust::reduce_by_key(
        thrust::device,
        keys.begin(),
        keys.end(),
        thrust::make_constant_iterator(1),
        unique.keys.begin(),
        unique.counts.begin());

    const size_t unique_size = static_cast<size_t>(ends.first - unique.keys.begin());
    unique.keys.resize(unique_size);
    unique.counts.resize(unique_size);
    return unique;
}

UniqueTileAccum reduce_tile_accum_by_key(thrust::device_vector<Key> keys,
                                         thrust::device_vector<TileAccumValue> values) {
    thrust::sort_by_key(thrust::device, keys.begin(), keys.end(), values.begin());

    UniqueTileAccum unique;
    unique.keys.resize(keys.size());
    unique.values.resize(values.size());
    auto ends = thrust::reduce_by_key(
        thrust::device,
        keys.begin(),
        keys.end(),
        values.begin(),
        unique.keys.begin(),
        unique.values.begin(),
        thrust::equal_to<Key>(),
        TileAccumPlus{});

    const size_t unique_size = static_cast<size_t>(ends.first - unique.keys.begin());
    unique.keys.resize(unique_size);
    unique.values.resize(unique_size);
    return unique;
}

template <typename KeyFn>
UniqueKeys reduce_sorted_transformed_keys(const thrust::device_vector<Key> &keys, KeyFn key_fn) {
    UniqueKeys unique;
    unique.keys.resize(keys.size());
    unique.counts.resize(keys.size());

    auto key_begin = thrust::make_transform_iterator(keys.begin(), key_fn);
    auto key_end = thrust::make_transform_iterator(keys.end(), key_fn);
    auto ends = thrust::reduce_by_key(
        thrust::device,
        key_begin,
        key_end,
        thrust::make_constant_iterator(1),
        unique.keys.begin(),
        unique.counts.begin());

    const size_t unique_size = static_cast<size_t>(ends.first - unique.keys.begin());
    unique.keys.resize(unique_size);
    unique.counts.resize(unique_size);
    return unique;
}

BaseTileAccumGpu build_base8_accum_gpu(const DeviceCsr &csr) {
    BaseTileAccumGpu base;
    base.rows = ceil_div_int(csr.rows, 8);
    base.cols = ceil_div_int(csr.cols, 8);
    if (csr.nnz == 0) {
        return base;
    }

    thrust::device_vector<Key> keys(static_cast<size_t>(csr.nnz));
    thrust::device_vector<TileAccumValue> values(static_cast<size_t>(csr.nnz));
    const int threads = 256;
    const int blocks = ceil_div_int(csr.rows, threads);
    fill_base8_accum_keys_kernel<<<blocks, threads>>>(
        csr.rows,
        thrust::raw_pointer_cast(csr.rowptr.data()),
        thrust::raw_pointer_cast(csr.colidx.data()),
        base.cols,
        thrust::raw_pointer_cast(keys.data()),
        thrust::raw_pointer_cast(values.data()));
    CUDA_CHECK(cudaGetLastError());

    UniqueTileAccum unique = reduce_tile_accum_by_key(std::move(keys), std::move(values));
    base.nnz = static_cast<int>(unique.keys.size());
    base.keys = std::move(unique.keys);
    base.values = std::move(unique.values);
    return base;
}

UniqueTileAccum merge_a_probe_shape(const BaseTileAccumGpu &base, int tile_m, int tile_n) {
    const int row_factor = tile_m / 8;
    const int col_factor = tile_n / 8;
    if (row_factor == 1 && col_factor == 1) {
        UniqueTileAccum unique;
        unique.keys = base.keys;
        unique.values = base.values;
        return unique;
    }

    thrust::device_vector<Key> keys(static_cast<size_t>(base.nnz));
    thrust::device_vector<TileAccumValue> values(static_cast<size_t>(base.nnz));
    if (base.nnz > 0) {
        const int threads = 256;
        const int blocks = ceil_div_int(base.nnz, threads);
        const int merged_cols = ceil_div_int(base.cols, col_factor);
        merge_a_probe_accum_keys_kernel<<<blocks, threads>>>(
            base.nnz,
            base.cols,
            thrust::raw_pointer_cast(base.keys.data()),
            thrust::raw_pointer_cast(base.values.data()),
            row_factor,
            col_factor,
            merged_cols,
            thrust::raw_pointer_cast(keys.data()),
            thrust::raw_pointer_cast(values.data()));
        CUDA_CHECK(cudaGetLastError());
    }
    return reduce_tile_accum_by_key(std::move(keys), std::move(values));
}

struct TileCsrGpu {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    thrust::device_vector<int> rowptr;
    thrust::device_vector<int> colidx;
    thrust::device_vector<float> values;
};

TileCsrGpu build_tile_csr_gpu(const DeviceCsr &csr, int tile_m, int tile_n, bool transpose) {
    TileCsrGpu out;
    out.rows = transpose ? ceil_div_int(csr.cols, tile_m) : ceil_div_int(csr.rows, tile_m);
    out.cols = transpose ? ceil_div_int(csr.rows, tile_n) : ceil_div_int(csr.cols, tile_n);

    thrust::device_vector<Key> keys(static_cast<size_t>(csr.nnz));
    const int threads = 256;
    const int blocks = ceil_div_int(csr.rows, threads);
    if (transpose) {
        fill_transposed_tile_keys_kernel<<<blocks, threads>>>(
            csr.rows,
            thrust::raw_pointer_cast(csr.rowptr.data()),
            thrust::raw_pointer_cast(csr.colidx.data()),
            tile_m,
            tile_n,
            out.cols,
            thrust::raw_pointer_cast(keys.data()));
    } else {
        fill_tile_keys_kernel<<<blocks, threads>>>(
            csr.rows,
            thrust::raw_pointer_cast(csr.rowptr.data()),
            thrust::raw_pointer_cast(csr.colidx.data()),
            tile_m,
            tile_n,
            out.cols,
            thrust::raw_pointer_cast(keys.data()),
            nullptr,
            nullptr);
    }
    CUDA_CHECK(cudaGetLastError());

    UniqueKeys unique = reduce_unique_keys(std::move(keys));
    out.nnz = static_cast<int>(unique.keys.size());

    out.rowptr.assign(static_cast<size_t>(out.rows + 1), 0);
    out.colidx.resize(static_cast<size_t>(out.nnz));
    out.values.assign(static_cast<size_t>(out.nnz), 1.0f);

    if (out.nnz > 0) {
        const int key_blocks = ceil_div_int(out.nnz, threads);
        count_tile_rows_kernel<<<key_blocks, threads>>>(
            thrust::raw_pointer_cast(unique.keys.data()),
            out.nnz,
            out.cols,
            thrust::raw_pointer_cast(out.rowptr.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::inclusive_scan(thrust::device, out.rowptr.begin(), out.rowptr.end(), out.rowptr.begin());
        keys_to_cols_kernel<<<key_blocks, threads>>>(
            thrust::raw_pointer_cast(unique.keys.data()),
            out.nnz,
            out.cols,
            thrust::raw_pointer_cast(out.colidx.data()));
        CUDA_CHECK(cudaGetLastError());
    }

    return out;
}

TileCsrGpu build_tile_csr_from_unique_keys(UniqueKeys unique, int rows, int cols) {
    TileCsrGpu out;
    out.rows = rows;
    out.cols = cols;
    out.nnz = static_cast<int>(unique.keys.size());
    out.rowptr.assign(static_cast<size_t>(out.rows + 1), 0);
    out.colidx.resize(static_cast<size_t>(out.nnz));
    out.values.assign(static_cast<size_t>(out.nnz), 1.0f);

    if (out.nnz > 0) {
        const int threads = 256;
        const int key_blocks = ceil_div_int(out.nnz, threads);
        count_tile_rows_kernel<<<key_blocks, threads>>>(
            thrust::raw_pointer_cast(unique.keys.data()),
            out.nnz,
            out.cols,
            thrust::raw_pointer_cast(out.rowptr.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::inclusive_scan(thrust::device, out.rowptr.begin(), out.rowptr.end(), out.rowptr.begin());
        keys_to_cols_kernel<<<key_blocks, threads>>>(
            thrust::raw_pointer_cast(unique.keys.data()),
            out.nnz,
            out.cols,
            thrust::raw_pointer_cast(out.colidx.data()));
        CUDA_CHECK(cudaGetLastError());
    }
    return out;
}

TileCsrGpu build_tile_csr_from_base_accum(const BaseTileAccumGpu &base) {
    TileCsrGpu out;
    out.rows = base.rows;
    out.cols = base.cols;
    out.nnz = base.nnz;
    out.rowptr.assign(static_cast<size_t>(out.rows + 1), 0);
    out.colidx.resize(static_cast<size_t>(out.nnz));
    out.values.assign(static_cast<size_t>(out.nnz), 1.0f);

    if (out.nnz > 0) {
        const int threads = 256;
        const int key_blocks = ceil_div_int(out.nnz, threads);
        count_tile_rows_kernel<<<key_blocks, threads>>>(
            thrust::raw_pointer_cast(base.keys.data()),
            out.nnz,
            out.cols,
            thrust::raw_pointer_cast(out.rowptr.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::inclusive_scan(thrust::device, out.rowptr.begin(), out.rowptr.end(), out.rowptr.begin());
        keys_to_cols_kernel<<<key_blocks, threads>>>(
            thrust::raw_pointer_cast(base.keys.data()),
            out.nnz,
            out.cols,
            thrust::raw_pointer_cast(out.colidx.data()));
        CUDA_CHECK(cudaGetLastError());
    }
    return out;
}

TileCsrGpu merge_tile_csr_gpu(const TileCsrGpu &base, int row_factor, int col_factor) {
    TileCsrGpu out;
    out.rows = ceil_div_int(base.rows, row_factor);
    out.cols = ceil_div_int(base.cols, col_factor);
    if (base.nnz == 0) {
        out.rowptr.assign(static_cast<size_t>(out.rows + 1), 0);
        return out;
    }

    thrust::device_vector<Key> keys(static_cast<size_t>(base.nnz));
    const int threads = 256;
    const int blocks = ceil_div_int(base.rows, threads);
    merge_tile_keys_kernel<<<blocks, threads>>>(
        base.rows,
        thrust::raw_pointer_cast(base.rowptr.data()),
        thrust::raw_pointer_cast(base.colidx.data()),
        row_factor,
        col_factor,
        out.cols,
        thrust::raw_pointer_cast(keys.data()));
    CUDA_CHECK(cudaGetLastError());

    UniqueKeys unique = reduce_unique_keys(std::move(keys));
    return build_tile_csr_from_unique_keys(std::move(unique), out.rows, out.cols);
}

struct SpGemmStats {
    long long nnz = 0;
    double avg = 0.0;
    int max = 0;
    int overflow_rows = 0;
    int fallback_rows = 0;
};

struct IsOverflowRow {
    const int *flags;

    __host__ __device__ bool operator()(int row) const {
        return flags[row] != 0;
    }
};

__global__ void c_matched_hash_kernel(int tile_rows,
                                      int b_tile_rows,
                                      const int *a_rowptr,
                                      const int *a_colidx,
                                      const int *b_rowptr,
                                      const int *b_colidx,
                                      unsigned long long *row_numblk,
                                      unsigned long long *row_matched_sum,
                                      int *row_max,
                                      int *row_overflow) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= tile_rows) {
        return;
    }

    __shared__ int keys[kCProbeHashCapacity];
    __shared__ int counts[kCProbeHashCapacity];
    __shared__ int overflow;

    for (int i = static_cast<int>(threadIdx.x); i < kCProbeHashCapacity; i += static_cast<int>(blockDim.x)) {
        keys[i] = -1;
        counts[i] = 0;
    }
    if (threadIdx.x == 0) {
        overflow = 0;
        row_numblk[row] = 0;
        row_matched_sum[row] = 0;
        row_max[row] = 0;
        row_overflow[row] = 0;
    }
    __syncthreads();

    const int a_begin = a_rowptr[row];
    const int a_end = a_rowptr[row + 1];
    for (int ap = a_begin; ap < a_end; ++ap) {
        const int k = a_colidx[ap];
        if (k < 0 || k >= b_tile_rows) {
            continue;
        }
        const int b_begin = b_rowptr[k];
        const int b_end = b_rowptr[k + 1];
        for (int bp = b_begin + static_cast<int>(threadIdx.x); bp < b_end; bp += static_cast<int>(blockDim.x)) {
            const int j = b_colidx[bp];
            unsigned int h = static_cast<unsigned int>(j) * 2654435761u;
            bool inserted = false;
            for (int probe = 0; probe < kCProbeHashCapacity; ++probe) {
                const int slot = static_cast<int>((h + static_cast<unsigned int>(probe)) &
                                                  static_cast<unsigned int>(kCProbeHashCapacity - 1));
                const int old = atomicCAS(&keys[slot], -1, j);
                if (old == -1 || old == j) {
                    atomicAdd(&counts[slot], 1);
                    inserted = true;
                    break;
                }
            }
            if (!inserted) {
                atomicExch(&overflow, 1);
            }
        }
    }
    __syncthreads();

    unsigned long long local_numblk = 0;
    unsigned long long local_sum = 0;
    int local_max = 0;
    for (int i = static_cast<int>(threadIdx.x); i < kCProbeHashCapacity; i += static_cast<int>(blockDim.x)) {
        const int cnt = counts[i];
        if (keys[i] >= 0 && cnt > 0) {
            local_numblk++;
            local_sum += cnt;
            local_max = local_max > cnt ? local_max : cnt;
        }
    }
    atomicAdd(&row_numblk[row], local_numblk);
    atomicAdd(&row_matched_sum[row], local_sum);
    atomicMax(&row_max[row], local_max);
    __syncthreads();

    if (threadIdx.x == 0 && overflow) {
        row_overflow[row] = 1;
    }
}

__global__ void c_matched_direct_kernel(int tile_rows,
                                        int b_tile_rows,
                                        int b_tile_cols,
                                        const int *a_rowptr,
                                        const int *a_colidx,
                                        const int *b_rowptr,
                                        const int *b_colidx,
                                        unsigned long long *row_numblk,
                                        unsigned long long *row_matched_sum,
                                        int *row_max,
                                        int *row_overflow) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= tile_rows) {
        return;
    }

    __shared__ int counts[kCProbeHashCapacity];

    for (int i = static_cast<int>(threadIdx.x); i < kCProbeHashCapacity; i += static_cast<int>(blockDim.x)) {
        counts[i] = 0;
    }
    if (threadIdx.x == 0) {
        row_numblk[row] = 0;
        row_matched_sum[row] = 0;
        row_max[row] = 0;
        row_overflow[row] = 0;
    }
    __syncthreads();

    const int a_begin = a_rowptr[row];
    const int a_end = a_rowptr[row + 1];
    for (int ap = a_begin; ap < a_end; ++ap) {
        const int k = a_colidx[ap];
        if (k < 0 || k >= b_tile_rows) {
            continue;
        }
        const int b_begin = b_rowptr[k];
        const int b_end = b_rowptr[k + 1];
        for (int bp = b_begin + static_cast<int>(threadIdx.x); bp < b_end; bp += static_cast<int>(blockDim.x)) {
            const int j = b_colidx[bp];
            if (j >= 0 && j < b_tile_cols) {
                atomicAdd(&counts[j], 1);
            }
        }
    }
    __syncthreads();

    unsigned long long local_numblk = 0;
    unsigned long long local_sum = 0;
    int local_max = 0;
    for (int i = static_cast<int>(threadIdx.x); i < b_tile_cols; i += static_cast<int>(blockDim.x)) {
        const int cnt = counts[i];
        if (cnt > 0) {
            local_numblk++;
            local_sum += cnt;
            local_max = local_max > cnt ? local_max : cnt;
        }
    }
    atomicAdd(&row_numblk[row], local_numblk);
    atomicAdd(&row_matched_sum[row], local_sum);
    atomicMax(&row_max[row], local_max);
}

__global__ void c_matched_hash_avg_kernel(int tile_rows,
                                          int b_tile_rows,
                                          const int *a_rowptr,
                                          const int *a_colidx,
                                          const int *b_rowptr,
                                          const int *b_colidx,
                                          unsigned long long *row_numblk,
                                          unsigned long long *row_matched_sum,
                                          int *row_overflow) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= tile_rows) {
        return;
    }

    __shared__ int keys[kCProbeHashCapacity];
    __shared__ int overflow;

    for (int i = static_cast<int>(threadIdx.x); i < kCProbeHashCapacity; i += static_cast<int>(blockDim.x)) {
        keys[i] = -1;
    }
    if (threadIdx.x == 0) {
        overflow = 0;
        row_numblk[row] = 0;
        row_matched_sum[row] = 0;
        row_overflow[row] = 0;
    }
    __syncthreads();

    unsigned long long local_sum = 0;
    const int a_begin = a_rowptr[row];
    const int a_end = a_rowptr[row + 1];
    for (int ap = a_begin; ap < a_end; ++ap) {
        const int k = a_colidx[ap];
        if (k < 0 || k >= b_tile_rows) {
            continue;
        }
        const int b_begin = b_rowptr[k];
        const int b_end = b_rowptr[k + 1];
        for (int bp = b_begin + static_cast<int>(threadIdx.x); bp < b_end; bp += static_cast<int>(blockDim.x)) {
            const int j = b_colidx[bp];
            local_sum++;
            unsigned int h = static_cast<unsigned int>(j) * 2654435761u;
            bool inserted = false;
            for (int probe = 0; probe < kCProbeHashCapacity; ++probe) {
                const int slot = static_cast<int>((h + static_cast<unsigned int>(probe)) &
                                                  static_cast<unsigned int>(kCProbeHashCapacity - 1));
                const int old = atomicCAS(&keys[slot], -1, j);
                if (old == -1 || old == j) {
                    inserted = true;
                    break;
                }
            }
            if (!inserted) {
                atomicExch(&overflow, 1);
            }
        }
    }
    atomicAdd(&row_matched_sum[row], local_sum);
    __syncthreads();

    unsigned long long local_numblk = 0;
    for (int i = static_cast<int>(threadIdx.x); i < kCProbeHashCapacity; i += static_cast<int>(blockDim.x)) {
        if (keys[i] >= 0) {
            local_numblk++;
        }
    }
    atomicAdd(&row_numblk[row], local_numblk);
    __syncthreads();

    if (threadIdx.x == 0 && overflow) {
        row_overflow[row] = 1;
    }
}

__global__ void c_matched_direct_avg_kernel(int tile_rows,
                                            int b_tile_rows,
                                            int b_tile_cols,
                                            const int *a_rowptr,
                                            const int *a_colidx,
                                            const int *b_rowptr,
                                            const int *b_colidx,
                                            unsigned long long *row_numblk,
                                            unsigned long long *row_matched_sum,
                                            int *row_overflow) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= tile_rows) {
        return;
    }

    __shared__ int flags[kCProbeHashCapacity];

    for (int i = static_cast<int>(threadIdx.x); i < kCProbeHashCapacity; i += static_cast<int>(blockDim.x)) {
        flags[i] = 0;
    }
    if (threadIdx.x == 0) {
        row_numblk[row] = 0;
        row_matched_sum[row] = 0;
        row_overflow[row] = 0;
    }
    __syncthreads();

    unsigned long long local_sum = 0;
    const int a_begin = a_rowptr[row];
    const int a_end = a_rowptr[row + 1];
    for (int ap = a_begin; ap < a_end; ++ap) {
        const int k = a_colidx[ap];
        if (k < 0 || k >= b_tile_rows) {
            continue;
        }
        const int b_begin = b_rowptr[k];
        const int b_end = b_rowptr[k + 1];
        for (int bp = b_begin + static_cast<int>(threadIdx.x); bp < b_end; bp += static_cast<int>(blockDim.x)) {
            const int j = b_colidx[bp];
            if (j >= 0 && j < b_tile_cols) {
                flags[j] = 1;
                local_sum++;
            }
        }
    }
    atomicAdd(&row_matched_sum[row], local_sum);
    __syncthreads();

    unsigned long long local_numblk = 0;
    for (int i = static_cast<int>(threadIdx.x); i < b_tile_cols; i += static_cast<int>(blockDim.x)) {
        if (flags[i]) {
            local_numblk++;
        }
    }
    atomicAdd(&row_numblk[row], local_numblk);
}

__global__ void clear_c_probe_rows_kernel(int num_rows,
                                          const int *rows,
                                          unsigned long long *row_numblk,
                                          unsigned long long *row_matched_sum,
                                          int *row_max,
                                          int *row_overflow) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (idx >= num_rows) {
        return;
    }
    const int row = rows[idx];
    row_numblk[row] = 0;
    row_matched_sum[row] = 0;
    row_max[row] = 0;
    row_overflow[row] = 0;
}

__global__ void c_matched_direct_segmented_kernel(int overflow_count,
                                                  const int *overflow_rows,
                                                  int b_tile_rows,
                                                  int b_tile_cols,
                                                  const int *a_rowptr,
                                                  const int *a_colidx,
                                                  const int *b_rowptr,
                                                  const int *b_colidx,
                                                  unsigned long long *row_numblk,
                                                  unsigned long long *row_matched_sum,
                                                  int *row_max,
                                                  int *row_overflow) {
    const int overflow_idx = static_cast<int>(blockIdx.x);
    const int segment = static_cast<int>(blockIdx.y);
    if (overflow_idx >= overflow_count) {
        return;
    }
    const int row = overflow_rows[overflow_idx];

    const int seg_begin = segment * kCProbeHashCapacity;
    const int seg_end = min(b_tile_cols, seg_begin + kCProbeHashCapacity);

    __shared__ int counts[kCProbeHashCapacity];

    for (int i = static_cast<int>(threadIdx.x); i < kCProbeHashCapacity; i += static_cast<int>(blockDim.x)) {
        counts[i] = 0;
    }
    __syncthreads();

    const int a_begin = a_rowptr[row];
    const int a_end = a_rowptr[row + 1];
    for (int ap = a_begin; ap < a_end; ++ap) {
        const int k = a_colidx[ap];
        if (k < 0 || k >= b_tile_rows) {
            continue;
        }
        const int b_begin = b_rowptr[k];
        const int b_end = b_rowptr[k + 1];
        for (int bp = b_begin + static_cast<int>(threadIdx.x); bp < b_end; bp += static_cast<int>(blockDim.x)) {
            const int j = b_colidx[bp];
            if (j < seg_begin || j >= seg_end) {
                continue;
            }
            atomicAdd(&counts[j - seg_begin], 1);
        }
    }
    __syncthreads();

    unsigned long long local_numblk = 0;
    unsigned long long local_sum = 0;
    int local_max = 0;
    const int segment_width = seg_end - seg_begin;
    for (int i = static_cast<int>(threadIdx.x); i < segment_width; i += static_cast<int>(blockDim.x)) {
        const int cnt = counts[i];
        if (cnt > 0) {
            local_numblk++;
            local_sum += cnt;
            local_max = local_max > cnt ? local_max : cnt;
        }
    }
    atomicAdd(&row_numblk[row], local_numblk);
    atomicAdd(&row_matched_sum[row], local_sum);
    atomicMax(&row_max[row], local_max);
    if (threadIdx.x == 0) {
        row_overflow[row] = 0;
    }
}

__global__ void c_matched_direct_avg_segmented_kernel(int overflow_count,
                                                      const int *overflow_rows,
                                                      int b_tile_rows,
                                                      int b_tile_cols,
                                                      const int *a_rowptr,
                                                      const int *a_colidx,
                                                      const int *b_rowptr,
                                                      const int *b_colidx,
                                                      unsigned long long *row_numblk,
                                                      unsigned long long *row_matched_sum,
                                                      int *row_overflow) {
    const int overflow_idx = static_cast<int>(blockIdx.x);
    const int segment = static_cast<int>(blockIdx.y);
    if (overflow_idx >= overflow_count) {
        return;
    }
    const int row = overflow_rows[overflow_idx];

    const int seg_begin = segment * kCProbeHashCapacity;
    const int seg_end = min(b_tile_cols, seg_begin + kCProbeHashCapacity);
    const int segment_width = seg_end - seg_begin;

    __shared__ int flags[kCProbeHashCapacity];

    for (int i = static_cast<int>(threadIdx.x); i < kCProbeHashCapacity; i += static_cast<int>(blockDim.x)) {
        flags[i] = 0;
    }
    __syncthreads();

    unsigned long long local_sum = 0;
    const int a_begin = a_rowptr[row];
    const int a_end = a_rowptr[row + 1];
    for (int ap = a_begin; ap < a_end; ++ap) {
        const int k = a_colidx[ap];
        if (k < 0 || k >= b_tile_rows) {
            continue;
        }
        const int b_begin = b_rowptr[k];
        const int b_end = b_rowptr[k + 1];
        for (int bp = b_begin + static_cast<int>(threadIdx.x); bp < b_end; bp += static_cast<int>(blockDim.x)) {
            const int j = b_colidx[bp];
            if (j < seg_begin || j >= seg_end) {
                continue;
            }
            flags[j - seg_begin] = 1;
            local_sum++;
        }
    }
    atomicAdd(&row_matched_sum[row], local_sum);
    __syncthreads();

    unsigned long long local_numblk = 0;
    for (int i = static_cast<int>(threadIdx.x); i < segment_width; i += static_cast<int>(blockDim.x)) {
        if (flags[i]) {
            local_numblk++;
        }
    }
    atomicAdd(&row_numblk[row], local_numblk);
    if (threadIdx.x == 0) {
        row_overflow[row] = 0;
    }
}

__global__ void c_matched_direct_avg_all_segmented_kernel(int tile_rows,
                                                          int b_tile_rows,
                                                          int b_tile_cols,
                                                          const int *a_rowptr,
                                                          const int *a_colidx,
                                                          const int *b_rowptr,
                                                          const int *b_colidx,
                                                          unsigned long long *row_numblk,
                                                          unsigned long long *row_matched_sum) {
    const int row = static_cast<int>(blockIdx.x);
    const int segment = static_cast<int>(blockIdx.y);
    if (row >= tile_rows) {
        return;
    }

    const int seg_begin = segment * kCProbeHashCapacity;
    const int seg_end = min(b_tile_cols, seg_begin + kCProbeHashCapacity);
    const int segment_width = seg_end - seg_begin;

    __shared__ int flags[kCProbeHashCapacity];

    for (int i = static_cast<int>(threadIdx.x); i < kCProbeHashCapacity; i += static_cast<int>(blockDim.x)) {
        flags[i] = 0;
    }
    __syncthreads();

    unsigned long long local_sum = 0;
    const int a_begin = a_rowptr[row];
    const int a_end = a_rowptr[row + 1];
    for (int ap = a_begin; ap < a_end; ++ap) {
        const int k = a_colidx[ap];
        if (k < 0 || k >= b_tile_rows) {
            continue;
        }
        const int b_begin = b_rowptr[k];
        const int b_end = b_rowptr[k + 1];
        for (int bp = b_begin + static_cast<int>(threadIdx.x); bp < b_end; bp += static_cast<int>(blockDim.x)) {
            const int j = b_colidx[bp];
            if (j < seg_begin || j >= seg_end) {
                continue;
            }
            flags[j - seg_begin] = 1;
            local_sum++;
        }
    }
    atomicAdd(&row_matched_sum[row], local_sum);
    __syncthreads();

    unsigned long long local_numblk = 0;
    for (int i = static_cast<int>(threadIdx.x); i < segment_width; i += static_cast<int>(blockDim.x)) {
        if (flags[i]) {
            local_numblk++;
        }
    }
    atomicAdd(&row_numblk[row], local_numblk);
}

SpGemmStats tile_spgemm_hash(const TileCsrGpu &a, const TileCsrGpu &b) {
    if (a.cols != b.rows) {
        throw std::runtime_error("tile CSR dimensions are incompatible for C probe");
    }

    thrust::device_vector<unsigned long long> row_numblk(static_cast<size_t>(a.rows), 0);
    thrust::device_vector<unsigned long long> row_matched_sum(static_cast<size_t>(a.rows), 0);
    thrust::device_vector<int> row_max(static_cast<size_t>(a.rows), 0);
    thrust::device_vector<int> row_overflow(static_cast<size_t>(a.rows), 0);

    auto reduce_stats = [&]() {
        SpGemmStats stats;
        const unsigned long long c_nnz = thrust::reduce(
            thrust::device,
            row_numblk.begin(),
            row_numblk.end(),
            0ULL,
            thrust::plus<unsigned long long>());
        const unsigned long long matched_sum =
            thrust::reduce(thrust::device,
                           row_matched_sum.begin(),
                           row_matched_sum.end(),
                           0ULL,
                           thrust::plus<unsigned long long>());
        stats.nnz = static_cast<long long>(c_nnz);
        stats.max = row_max.empty() ? 0 : *thrust::max_element(thrust::device, row_max.begin(), row_max.end());
        stats.overflow_rows =
            thrust::reduce(thrust::device, row_overflow.begin(), row_overflow.end(), 0, thrust::plus<int>());
        stats.avg = stats.nnz ? static_cast<double>(matched_sum) / static_cast<double>(stats.nnz) : 0.0;
        return stats;
    };

    if (a.rows > 0) {
        if (b.cols <= kCProbeHashCapacity) {
            c_matched_direct_kernel<<<a.rows, 256>>>(
                a.rows,
                b.rows,
                b.cols,
                thrust::raw_pointer_cast(a.rowptr.data()),
                thrust::raw_pointer_cast(a.colidx.data()),
                thrust::raw_pointer_cast(b.rowptr.data()),
                thrust::raw_pointer_cast(b.colidx.data()),
                thrust::raw_pointer_cast(row_numblk.data()),
                thrust::raw_pointer_cast(row_matched_sum.data()),
                thrust::raw_pointer_cast(row_max.data()),
                thrust::raw_pointer_cast(row_overflow.data()));
        } else {
            c_matched_hash_kernel<<<a.rows, 256>>>(
                a.rows,
                b.rows,
                thrust::raw_pointer_cast(a.rowptr.data()),
                thrust::raw_pointer_cast(a.colidx.data()),
                thrust::raw_pointer_cast(b.rowptr.data()),
                thrust::raw_pointer_cast(b.colidx.data()),
                thrust::raw_pointer_cast(row_numblk.data()),
                thrust::raw_pointer_cast(row_matched_sum.data()),
                thrust::raw_pointer_cast(row_max.data()),
                thrust::raw_pointer_cast(row_overflow.data()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    SpGemmStats stats = reduce_stats();
    if (stats.overflow_rows > 0) {
        thrust::device_vector<int> overflow_rows(static_cast<size_t>(stats.overflow_rows));
        auto overflow_end = thrust::copy_if(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(a.rows),
            overflow_rows.begin(),
            IsOverflowRow{thrust::raw_pointer_cast(row_overflow.data())});
        const int overflow_count = static_cast<int>(overflow_end - overflow_rows.begin());
        if (overflow_count != stats.overflow_rows) {
            overflow_rows.resize(static_cast<size_t>(overflow_count));
        }
        stats.fallback_rows += overflow_count;

        const int clear_threads = 256;
        const int clear_blocks = ceil_div_int(overflow_count, clear_threads);
        if (overflow_count > 0) {
            clear_c_probe_rows_kernel<<<clear_blocks, clear_threads>>>(
                overflow_count,
                thrust::raw_pointer_cast(overflow_rows.data()),
                thrust::raw_pointer_cast(row_numblk.data()),
                thrust::raw_pointer_cast(row_matched_sum.data()),
                thrust::raw_pointer_cast(row_max.data()),
                thrust::raw_pointer_cast(row_overflow.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        const int segments = ceil_div_int(b.cols, kCProbeHashCapacity);
        if (overflow_count > 0 && segments > 0) {
            const dim3 grid(static_cast<unsigned int>(overflow_count), static_cast<unsigned int>(segments));
            c_matched_direct_segmented_kernel<<<grid, 256>>>(
                overflow_count,
                thrust::raw_pointer_cast(overflow_rows.data()),
                b.rows,
                b.cols,
                thrust::raw_pointer_cast(a.rowptr.data()),
                thrust::raw_pointer_cast(a.colidx.data()),
                thrust::raw_pointer_cast(b.rowptr.data()),
                thrust::raw_pointer_cast(b.colidx.data()),
                thrust::raw_pointer_cast(row_numblk.data()),
                thrust::raw_pointer_cast(row_matched_sum.data()),
                thrust::raw_pointer_cast(row_max.data()),
                thrust::raw_pointer_cast(row_overflow.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        const int fallback_rows = stats.fallback_rows;
        stats = reduce_stats();
        stats.fallback_rows = fallback_rows;
    }
    return stats;
}

SpGemmStats tile_spgemm_hash_avg(const TileCsrGpu &a, const TileCsrGpu &b) {
    if (a.cols != b.rows) {
        throw std::runtime_error("tile CSR dimensions are incompatible for C probe");
    }

    thrust::device_vector<unsigned long long> row_numblk(static_cast<size_t>(a.rows), 0);
    thrust::device_vector<unsigned long long> row_matched_sum(static_cast<size_t>(a.rows), 0);
    thrust::device_vector<int> row_max(static_cast<size_t>(a.rows), 0);
    thrust::device_vector<int> row_overflow(static_cast<size_t>(a.rows), 0);

    auto reduce_stats = [&]() {
        SpGemmStats stats;
        const unsigned long long c_nnz = thrust::reduce(
            thrust::device,
            row_numblk.begin(),
            row_numblk.end(),
            0ULL,
            thrust::plus<unsigned long long>());
        const unsigned long long matched_sum =
            thrust::reduce(thrust::device,
                           row_matched_sum.begin(),
                           row_matched_sum.end(),
                           0ULL,
                           thrust::plus<unsigned long long>());
        stats.nnz = static_cast<long long>(c_nnz);
        stats.max = -1;
        stats.overflow_rows =
            thrust::reduce(thrust::device, row_overflow.begin(), row_overflow.end(), 0, thrust::plus<int>());
        stats.avg = stats.nnz ? static_cast<double>(matched_sum) / static_cast<double>(stats.nnz) : 0.0;
        return stats;
    };

    if (a.rows > 0) {
        if (b.cols <= kCProbeHashCapacity) {
            c_matched_direct_avg_kernel<<<a.rows, 256>>>(
                a.rows,
                b.rows,
                b.cols,
                thrust::raw_pointer_cast(a.rowptr.data()),
                thrust::raw_pointer_cast(a.colidx.data()),
                thrust::raw_pointer_cast(b.rowptr.data()),
                thrust::raw_pointer_cast(b.colidx.data()),
                thrust::raw_pointer_cast(row_numblk.data()),
                thrust::raw_pointer_cast(row_matched_sum.data()),
                thrust::raw_pointer_cast(row_overflow.data()));
        } else {
            const int segments = ceil_div_int(b.cols, kCProbeHashCapacity);
            if (segments <= kCProbeAvgAllSegmentLimit) {
                const dim3 grid(static_cast<unsigned int>(a.rows), static_cast<unsigned int>(segments));
                c_matched_direct_avg_all_segmented_kernel<<<grid, 256>>>(
                    a.rows,
                    b.rows,
                    b.cols,
                    thrust::raw_pointer_cast(a.rowptr.data()),
                    thrust::raw_pointer_cast(a.colidx.data()),
                    thrust::raw_pointer_cast(b.rowptr.data()),
                    thrust::raw_pointer_cast(b.colidx.data()),
                    thrust::raw_pointer_cast(row_numblk.data()),
                    thrust::raw_pointer_cast(row_matched_sum.data()));
            } else {
                c_matched_hash_avg_kernel<<<a.rows, 256>>>(
                    a.rows,
                    b.rows,
                    thrust::raw_pointer_cast(a.rowptr.data()),
                    thrust::raw_pointer_cast(a.colidx.data()),
                    thrust::raw_pointer_cast(b.rowptr.data()),
                    thrust::raw_pointer_cast(b.colidx.data()),
                    thrust::raw_pointer_cast(row_numblk.data()),
                    thrust::raw_pointer_cast(row_matched_sum.data()),
                    thrust::raw_pointer_cast(row_overflow.data()));
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    SpGemmStats stats = reduce_stats();
    if (stats.overflow_rows > 0) {
        thrust::device_vector<int> overflow_rows(static_cast<size_t>(stats.overflow_rows));
        auto overflow_end = thrust::copy_if(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(a.rows),
            overflow_rows.begin(),
            IsOverflowRow{thrust::raw_pointer_cast(row_overflow.data())});
        const int overflow_count = static_cast<int>(overflow_end - overflow_rows.begin());
        if (overflow_count != stats.overflow_rows) {
            overflow_rows.resize(static_cast<size_t>(overflow_count));
        }
        stats.fallback_rows += overflow_count;

        const int clear_threads = 256;
        const int clear_blocks = ceil_div_int(overflow_count, clear_threads);
        if (overflow_count > 0) {
            clear_c_probe_rows_kernel<<<clear_blocks, clear_threads>>>(
                overflow_count,
                thrust::raw_pointer_cast(overflow_rows.data()),
                thrust::raw_pointer_cast(row_numblk.data()),
                thrust::raw_pointer_cast(row_matched_sum.data()),
                thrust::raw_pointer_cast(row_max.data()),
                thrust::raw_pointer_cast(row_overflow.data()));
            CUDA_CHECK(cudaGetLastError());
        }

        const int segments = ceil_div_int(b.cols, kCProbeHashCapacity);
        if (overflow_count > 0 && segments > 0) {
            const dim3 grid(static_cast<unsigned int>(overflow_count), static_cast<unsigned int>(segments));
            c_matched_direct_avg_segmented_kernel<<<grid, 256>>>(
                overflow_count,
                thrust::raw_pointer_cast(overflow_rows.data()),
                b.rows,
                b.cols,
                thrust::raw_pointer_cast(a.rowptr.data()),
                thrust::raw_pointer_cast(a.colidx.data()),
                thrust::raw_pointer_cast(b.rowptr.data()),
                thrust::raw_pointer_cast(b.colidx.data()),
                thrust::raw_pointer_cast(row_numblk.data()),
                thrust::raw_pointer_cast(row_matched_sum.data()),
                thrust::raw_pointer_cast(row_overflow.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        const int fallback_rows = stats.fallback_rows;
        stats = reduce_stats();
        stats.fallback_rows = fallback_rows;
    }
    return stats;
}

SpGemmStats tile_spgemm_cusparse(cusparseHandle_t handle, const TileCsrGpu &a, const TileCsrGpu &b) {
    if (a.cols != b.rows) {
        throw std::runtime_error("tile CSR dimensions are incompatible for C probe");
    }

    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseSpMatDescr_t mat_b = nullptr;
    cusparseSpMatDescr_t mat_c = nullptr;

    int *d_c_rowptr = nullptr;
    int *d_c_colidx = nullptr;
    float *d_c_values = nullptr;
    void *d_buffer1 = nullptr;
    void *d_buffer2 = nullptr;
    void *d_buffer3 = nullptr;
    size_t buffer_size1 = 0;
    size_t buffer_size2 = 0;
    size_t buffer_size3 = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    const cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_ALG2;
    const float chunk_fraction = 0.01f;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_rowptr), static_cast<size_t>(a.rows + 1) * sizeof(int)));

    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_a,
        a.rows,
        a.cols,
        a.nnz,
        const_cast<int *>(thrust::raw_pointer_cast(a.rowptr.data())),
        const_cast<int *>(thrust::raw_pointer_cast(a.colidx.data())),
        const_cast<float *>(thrust::raw_pointer_cast(a.values.data())),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_b,
        b.rows,
        b.cols,
        b.nnz,
        const_cast<int *>(thrust::raw_pointer_cast(b.rowptr.data())),
        const_cast<int *>(thrust::raw_pointer_cast(b.colidx.data())),
        const_cast<float *>(thrust::raw_pointer_cast(b.values.data())),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_c,
        a.rows,
        b.cols,
        0,
        d_c_rowptr,
        nullptr,
        nullptr,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    cusparseSpGEMMDescr_t desc = nullptr;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&desc));

    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        mat_a,
        mat_b,
        &beta,
        mat_c,
        CUDA_R_32F,
        alg,
        desc,
        &buffer_size1,
        nullptr));
    if (buffer_size1 > 0) {
        CUDA_CHECK(cudaMalloc(&d_buffer1, buffer_size1));
    }
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        mat_a,
        mat_b,
        &beta,
        mat_c,
        CUDA_R_32F,
        alg,
        desc,
        &buffer_size1,
        d_buffer1));

    CUSPARSE_CHECK(cusparseSpGEMM_estimateMemory(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        mat_a,
        mat_b,
        &beta,
        mat_c,
        CUDA_R_32F,
        alg,
        desc,
        chunk_fraction,
        &buffer_size3,
        nullptr,
        &buffer_size2));
    if (buffer_size3 > 0) {
        CUDA_CHECK(cudaMalloc(&d_buffer3, buffer_size3));
    }
    CUSPARSE_CHECK(cusparseSpGEMM_estimateMemory(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        mat_a,
        mat_b,
        &beta,
        mat_c,
        CUDA_R_32F,
        alg,
        desc,
        chunk_fraction,
        &buffer_size3,
        d_buffer3,
        &buffer_size2));
    if (buffer_size2 > 0) {
        CUDA_CHECK(cudaMalloc(&d_buffer2, buffer_size2));
    }
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        mat_a,
        mat_b,
        &beta,
        mat_c,
        CUDA_R_32F,
        alg,
        desc,
        &buffer_size2,
        d_buffer2));

    int64_t c_rows = 0;
    int64_t c_cols = 0;
    int64_t c_nnz = 0;
    CUSPARSE_CHECK(cusparseSpMatGetSize(mat_c, &c_rows, &c_cols, &c_nnz));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_colidx), static_cast<size_t>(c_nnz) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_values), static_cast<size_t>(c_nnz) * sizeof(float)));
    CUSPARSE_CHECK(cusparseCsrSetPointers(mat_c, d_c_rowptr, d_c_colidx, d_c_values));

    CUSPARSE_CHECK(cusparseSpGEMM_copy(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        mat_a,
        mat_b,
        &beta,
        mat_c,
        CUDA_R_32F,
        alg,
        desc));

    SpGemmStats stats;
    stats.nnz = static_cast<long long>(c_nnz);
    if (c_nnz > 0) {
        thrust::device_ptr<float> vals(d_c_values);
        const float sum = thrust::reduce(thrust::device, vals, vals + c_nnz, 0.0f, thrust::plus<float>());
        const float max_val = *thrust::max_element(thrust::device, vals, vals + c_nnz);
        stats.avg = static_cast<double>(sum) / static_cast<double>(c_nnz);
        stats.max = static_cast<int>(max_val + 0.5f);
    }

    if (desc) CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(desc));
    if (mat_a) CUSPARSE_CHECK(cusparseDestroySpMat(mat_a));
    if (mat_b) CUSPARSE_CHECK(cusparseDestroySpMat(mat_b));
    if (mat_c) CUSPARSE_CHECK(cusparseDestroySpMat(mat_c));
    cudaFree(d_buffer1);
    cudaFree(d_buffer2);
    cudaFree(d_buffer3);
    cudaFree(d_c_rowptr);
    cudaFree(d_c_colidx);
    cudaFree(d_c_values);

    return stats;
}

void run_a_probe_gpu_sort3(const DeviceCsr &csr, MatrixResult *result) {
    double ms = 0.0;
    long long tiles_16x16 = 0;
    ms = timed_cuda_ms([&]() {
        const int threads = 256;
        const int blocks = ceil_div_int(csr.rows, threads);
        for (const auto &tile_size : kAProbeTileSizes) {
            const int tile_m = tile_size[0];
            const int tile_n = tile_size[1];
            const int tile_cols = ceil_div_int(csr.cols, tile_n);

            thrust::device_vector<Key> tile_keys(static_cast<size_t>(csr.nnz));
            thrust::device_vector<Key> row_slot_keys(static_cast<size_t>(csr.nnz));
            thrust::device_vector<Key> col_slot_keys(static_cast<size_t>(csr.nnz));
            fill_tile_keys_kernel<<<blocks, threads>>>(
                csr.rows,
                thrust::raw_pointer_cast(csr.rowptr.data()),
                thrust::raw_pointer_cast(csr.colidx.data()),
                tile_m,
                tile_n,
                tile_cols,
                thrust::raw_pointer_cast(tile_keys.data()),
                thrust::raw_pointer_cast(row_slot_keys.data()),
                thrust::raw_pointer_cast(col_slot_keys.data()));
            CUDA_CHECK(cudaGetLastError());

            UniqueKeys tile_unique = reduce_unique_keys(std::move(tile_keys));
            UniqueKeys row_unique = reduce_unique_keys(std::move(row_slot_keys));
            UniqueKeys col_unique = reduce_unique_keys(std::move(col_slot_keys));
            if (tile_m == 16 && tile_n == 16) {
                tiles_16x16 = static_cast<long long>(tile_unique.keys.size());
            }

            // Keep the reductions live so an optimizing compiler cannot remove work.
            volatile long long sink = static_cast<long long>(tile_unique.keys.size() +
                                                             row_unique.keys.size() +
                                                             col_unique.keys.size());
            (void)sink;
        }
    });

    result->a_probe_ms = ms;
    result->a_tiles_16x16 = tiles_16x16;
}

void run_a_probe_gpu_packed(const DeviceCsr &csr, MatrixResult *result) {
    double ms = 0.0;
    long long tiles_16x16 = 0;
    ms = timed_cuda_ms([&]() {
        const int threads = 256;
        const int blocks = ceil_div_int(csr.rows, threads);
        for (const auto &tile_size : kAProbeTileSizes) {
            const int tile_m = tile_size[0];
            const int tile_n = tile_size[1];
            const int tile_cols = ceil_div_int(csr.cols, tile_n);

            thrust::device_vector<Key> row_major_keys(static_cast<size_t>(csr.nnz));
            thrust::device_vector<Key> col_major_keys(static_cast<size_t>(csr.nnz));
            fill_packed_a_probe_keys_kernel<<<blocks, threads>>>(
                csr.rows,
                thrust::raw_pointer_cast(csr.rowptr.data()),
                thrust::raw_pointer_cast(csr.colidx.data()),
                tile_m,
                tile_n,
                tile_cols,
                thrust::raw_pointer_cast(row_major_keys.data()),
                thrust::raw_pointer_cast(col_major_keys.data()));
            CUDA_CHECK(cudaGetLastError());

            thrust::sort(thrust::device, row_major_keys.begin(), row_major_keys.end());
            UniqueKeys tile_unique = reduce_sorted_transformed_keys(row_major_keys, PackedTileKey{});
            UniqueKeys row_unique = reduce_sorted_transformed_keys(row_major_keys, PackedMajorSlotKey{});

            thrust::sort(thrust::device, col_major_keys.begin(), col_major_keys.end());
            UniqueKeys col_unique = reduce_sorted_transformed_keys(col_major_keys, PackedMajorSlotKey{});

            if (tile_m == 16 && tile_n == 16) {
                tiles_16x16 = static_cast<long long>(tile_unique.keys.size());
            }

            volatile long long sink = static_cast<long long>(tile_unique.keys.size() +
                                                             row_unique.keys.size() +
                                                             col_unique.keys.size());
            (void)sink;
        }
    });

    result->a_probe_ms = ms;
    result->a_tiles_16x16 = tiles_16x16;
}

void consume_a_probe_merge8_shape(const BaseTileAccumGpu &base,
                                  int tile_m,
                                  int tile_n,
                                  long long *tiles_16x16) {
    const bool is_base_shape = (tile_m == 8 && tile_n == 8);
    UniqueTileAccum unique;
    const thrust::device_vector<TileAccumValue> *values = nullptr;
    size_t num_tiles = 0;
    if (is_base_shape) {
        values = &base.values;
        num_tiles = static_cast<size_t>(base.nnz);
    } else {
        unique = merge_a_probe_shape(base, tile_m, tile_n);
        values = &unique.values;
        num_tiles = unique.keys.size();
    }

    const unsigned long long row_slots = thrust::transform_reduce(
        thrust::device,
        values->begin(),
        values->end(),
        RowSlotCount{},
        0ULL,
        thrust::plus<unsigned long long>());
    const unsigned long long col_slots = thrust::transform_reduce(
        thrust::device,
        values->begin(),
        values->end(),
        ColSlotCount{},
        0ULL,
        thrust::plus<unsigned long long>());

    if (tile_m == 16 && tile_n == 16) {
        *tiles_16x16 = static_cast<long long>(num_tiles);
    }

    volatile unsigned long long sink =
        static_cast<unsigned long long>(num_tiles) + row_slots + col_slots;
    (void)sink;
}

void run_a_probe_gpu_merge8(const DeviceCsr &csr, MatrixResult *result, BaseTileAccumGpu *shared_base) {
    double ms = 0.0;
    long long tiles_16x16 = 0;
    ms = timed_cuda_ms([&]() {
        BaseTileAccumGpu base = build_base8_accum_gpu(csr);
        for (const auto &tile_size : kAProbeTileSizes) {
            const int tile_m = tile_size[0];
            const int tile_n = tile_size[1];
            consume_a_probe_merge8_shape(base, tile_m, tile_n, &tiles_16x16);
        }
        if (shared_base) {
            *shared_base = std::move(base);
        }
    });

    result->a_probe_ms = ms;
    result->a_tiles_16x16 = tiles_16x16;
}

void run_a_probe_gpu(const DeviceCsr &csr,
                     AProbeImpl impl,
                     MatrixResult *result,
                     BaseTileAccumGpu *shared_base = nullptr) {
    if (impl == AProbeImpl::Merge8) {
        run_a_probe_gpu_merge8(csr, result, shared_base);
    } else if (impl == AProbeImpl::Packed) {
        run_a_probe_gpu_packed(csr, result);
    } else {
        run_a_probe_gpu_sort3(csr, result);
    }
}

void run_c_probe_gpu(const DeviceCsr &csr,
                     bool aat,
                     CProbeImpl impl,
                     const BaseTileAccumGpu *shared_base,
                     MatrixResult *result) {
    cusparseHandle_t handle = nullptr;
    if (impl == CProbeImpl::Cusparse) {
        CUSPARSE_CHECK(cusparseCreate(&handle));
    }

    double build_ms_total = 0.0;
    double feature_ms_total = 0.0;
    std::array<TileCsrGpu, 3> tiles_a;
    std::array<TileCsrGpu, 3> tiles_b;

    build_ms_total = timed_cuda_ms([&]() {
        if (shared_base) {
            tiles_a[0] = build_tile_csr_from_base_accum(*shared_base);
        } else {
            tiles_a[0] = build_tile_csr_gpu(csr, 8, 8, false);
        }
        tiles_a[1] = merge_tile_csr_gpu(tiles_a[0], 2, 2);
        tiles_a[2] = merge_tile_csr_gpu(tiles_a[0], 4, 4);
        if (aat) {
            tiles_b[0] = build_tile_csr_gpu(csr, 8, 8, true);
            tiles_b[1] = merge_tile_csr_gpu(tiles_b[0], 2, 2);
            tiles_b[2] = merge_tile_csr_gpu(tiles_b[0], 4, 4);
        }
    });

    for (size_t shape = 0; shape < sizeof(kCProbeTileMs) / sizeof(kCProbeTileMs[0]); ++shape) {
        const int tile_m = kCProbeTileMs[shape];
        const TileCsrGpu &tile_a = tiles_a[shape];
        const TileCsrGpu &tile_b_ref = aat ? tiles_b[shape] : tiles_a[shape];

        SpGemmStats stats;
        const double feature_ms = timed_cuda_ms([&]() {
            if (impl == CProbeImpl::Hash) {
                stats = tile_spgemm_hash(tile_a, tile_b_ref);
            } else if (impl == CProbeImpl::HashAvg) {
                stats = tile_spgemm_hash_avg(tile_a, tile_b_ref);
            } else {
                stats = tile_spgemm_cusparse(handle, tile_a, tile_b_ref);
            }
        });
        feature_ms_total += feature_ms;
        result->c_hash_overflow_rows += stats.overflow_rows;
        result->c_hash_fallback_rows += stats.fallback_rows;
        if (stats.overflow_rows > 0) {
            std::ostringstream os;
            os << "GPU C probe hash table overflow on tile_m=" << tile_m
               << " for " << stats.overflow_rows << " tile rows"
               << " (increase kCProbeHashCapacity or use --c-probe-impl cusparse)";
            throw std::runtime_error(os.str());
        }

        if (tile_m == 8) {
            result->c_tiles_8 = stats.nnz;
            result->c_avg_8 = stats.avg;
            result->c_max_8 = stats.max;
        } else if (tile_m == 16) {
            result->c_tiles_16 = stats.nnz;
            result->c_avg_16 = stats.avg;
            result->c_max_16 = stats.max;
        } else if (tile_m == 32) {
            result->c_tiles_32 = stats.nnz;
            result->c_avg_32 = stats.avg;
            result->c_max_32 = stats.max;
        }
    }

    if (handle) {
        CUSPARSE_CHECK(cusparseDestroy(handle));
    }
    result->c_build_ms = build_ms_total;
    result->c_feature_ms = feature_ms_total;
}

void run_csr2tile_pattern_gpu(const DeviceCsr &csr, MatrixResult *result) {
    long long num_tiles = 0;
    const double ms = timed_cuda_ms([&]() {
        TileCsrGpu tile = build_tile_csr_gpu(csr, 16, 16, false);
        num_tiles = tile.nnz;
    });
    result->csr2tile_pattern_ms = ms;
    result->csr2tile_tiles = num_tiles;
}

DeviceCsr copy_to_device(const bench::CsrMatrix &matrix, double *h2d_ms) {
    DeviceCsr csr;
    csr.rows = matrix.rows;
    csr.cols = matrix.cols;
    csr.nnz = static_cast<int>(matrix.nnz);

    bench::Timer timer;
    csr.rowptr = matrix.rowptr;
    csr.colidx = matrix.colidx;
    CUDA_CHECK(cudaDeviceSynchronize());
    *h2d_ms = timer.elapsed_ms();
    return csr;
}

void print_csv_header() {
    std::cout
        << "matrix,rows,cols,nnz,symmetric,load_ms,h2d_ms,"
        << "gpu_a_probe_ms,gpu_a_tiles_16x16,"
        << "gpu_c_build_ms,gpu_c_feature_ms,gpu_c_total_ms,"
        << "gpu_c_tiles_8,gpu_c_avg_8,gpu_c_max_8,"
        << "gpu_c_tiles_16,gpu_c_avg_16,gpu_c_max_16,"
        << "gpu_c_tiles_32,gpu_c_avg_32,gpu_c_max_32,"
        << "gpu_c_hash_overflow_rows,gpu_c_hash_fallback_rows,"
        << "gpu_csr2tile_pattern_ms,gpu_csr2tile_tiles\n";
}

void print_csv_row(const MatrixResult &r) {
    std::cout << r.matrix << ','
              << r.rows << ','
              << r.cols << ','
              << r.nnz << ','
              << r.symmetric << ','
              << std::fixed << std::setprecision(4)
              << r.load_ms << ','
              << r.h2d_ms << ','
              << r.a_probe_ms << ','
              << r.a_tiles_16x16 << ','
              << r.c_build_ms << ','
              << r.c_feature_ms << ','
              << (r.c_build_ms + r.c_feature_ms) << ','
              << r.c_tiles_8 << ','
              << r.c_avg_8 << ','
              << r.c_max_8 << ','
              << r.c_tiles_16 << ','
              << r.c_avg_16 << ','
              << r.c_max_16 << ','
              << r.c_tiles_32 << ','
              << r.c_avg_32 << ','
              << r.c_max_32 << ','
              << r.c_hash_overflow_rows << ','
              << r.c_hash_fallback_rows << ','
              << r.csr2tile_pattern_ms << ','
              << r.csr2tile_tiles << '\n';
}

void usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " [--device N] [--aat] [--csv] [--skip-c-probe]"
              << " [--a-probe-impl merge8|packed|sort3]"
              << " [--c-probe-impl hash|hash-avg|cusparse] <matrix.mtx>...\n";
}

}  // namespace

int main(int argc, char **argv) {
    int device = 0;
    bool aat = false;
    bool csv = false;
    bool skip_c_probe = false;
    AProbeImpl a_probe_impl = AProbeImpl::Merge8;
    CProbeImpl c_probe_impl = CProbeImpl::HashAvg;
    std::vector<std::string> matrix_paths;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            device = std::atoi(argv[i]);
        } else if (arg == "--aat") {
            aat = true;
        } else if (arg == "--csv") {
            csv = true;
        } else if (arg == "--skip-c-probe") {
            skip_c_probe = true;
        } else if (arg == "--a-probe-impl") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            const std::string value = argv[i];
            if (value == "merge8") {
                a_probe_impl = AProbeImpl::Merge8;
            } else if (value == "packed") {
                a_probe_impl = AProbeImpl::Packed;
            } else if (value == "sort3") {
                a_probe_impl = AProbeImpl::Sort3;
            } else {
                usage(argv[0]);
                return 1;
            }
        } else if (arg == "--c-probe-impl") {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            const std::string value = argv[i];
            if (value == "hash") {
                c_probe_impl = CProbeImpl::Hash;
            } else if (value == "hash-avg") {
                c_probe_impl = CProbeImpl::HashAvg;
            } else if (value == "cusparse") {
                c_probe_impl = CProbeImpl::Cusparse;
            } else {
                usage(argv[0]);
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            matrix_paths.push_back(arg);
        }
    }

    if (matrix_paths.empty()) {
        usage(argv[0]);
        return 1;
    }

    try {
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        if (!csv) {
            std::cout << "GPU device " << device << ": " << prop.name << "\n";
            std::cout << "A probe impl: " << a_probe_impl_name(a_probe_impl) << "\n";
            std::cout << "C probe mode: " << (aat ? "A*A^T" : "A*A") << "\n";
            std::cout << "C probe impl: " << c_probe_impl_name(c_probe_impl) << "\n";
            if (skip_c_probe) {
                std::cout << "C probe skipped\n";
            }
        } else {
            print_csv_header();
        }

        for (const std::string &path : matrix_paths) {
            MatrixResult result;
            result.matrix = bench::basename(path);

            bench::Timer load_timer;
            bench::CsrMatrix matrix = bench::load_matrix_market(path);
            result.load_ms = load_timer.elapsed_ms();
            result.rows = matrix.rows;
            result.cols = matrix.cols;
            result.nnz = matrix.nnz;
            result.symmetric = matrix.symmetric ? 1 : 0;

            if (!aat && matrix.rows != matrix.cols) {
                throw std::runtime_error("A*A mode requires square matrices: " + path);
            }

            DeviceCsr csr = copy_to_device(matrix, &result.h2d_ms);
            BaseTileAccumGpu shared_base8;
            BaseTileAccumGpu *shared_base8_ptr = (a_probe_impl == AProbeImpl::Merge8) ? &shared_base8 : nullptr;
            run_a_probe_gpu(csr, a_probe_impl, &result, shared_base8_ptr);
            if (!skip_c_probe) {
                run_c_probe_gpu(csr, aat, c_probe_impl, shared_base8_ptr, &result);
            }
            run_csr2tile_pattern_gpu(csr, &result);

            if (csv) {
                print_csv_row(result);
            } else {
                std::cout << "\n[" << result.matrix << "] "
                          << result.rows << "x" << result.cols
                          << ", nnz=" << result.nnz << "\n";
                std::cout << "  load_ms                 : " << result.load_ms << "\n";
                std::cout << "  h2d_ms                  : " << result.h2d_ms << "\n";
                std::cout << "  gpu_a_probe_ms          : " << result.a_probe_ms
                          << " (16x16 tiles=" << result.a_tiles_16x16 << ")\n";
                std::cout << "  gpu_c_build_ms          : " << result.c_build_ms << "\n";
                std::cout << "  gpu_c_feature_ms        : " << result.c_feature_ms << "\n";
                std::cout << "  gpu_c_total_ms          : " << (result.c_build_ms + result.c_feature_ms) << "\n";
                std::cout << "  gpu_c_16                : tiles=" << result.c_tiles_16
                          << ", avg=" << result.c_avg_16
                          << ", max=" << result.c_max_16 << "\n";
                std::cout << "  gpu_c_hash_overflow_rows: " << result.c_hash_overflow_rows << "\n";
                std::cout << "  gpu_c_hash_fallback_rows: " << result.c_hash_fallback_rows << "\n";
                std::cout << "  gpu_csr2tile_pattern_ms : " << result.csr2tile_pattern_ms
                          << " (tiles=" << result.csr2tile_tiles << ")\n";
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
